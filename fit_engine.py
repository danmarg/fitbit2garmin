"""
fit_engine.py
Encodes Garmin Monitoring FIT files (file type 9) from Google Health (Fitbit) intraday data.
Uses the Garmin FIT SDK (Python) to produce correctly-checksummed binary files.
"""

import io
import struct
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Garmin FIT protocol constants
# ---------------------------------------------------------------------------

FIT_PROTOCOL_VERSION = 0x10       # Protocol version 1.0
FIT_PROFILE_VERSION = 2049        # Profile 20.49 (matches real Forerunner firmware)
GARMIN_EPOCH = 631065600          # Unix timestamp of 1989-12-31 00:00:00 UTC

# Global message numbers (from FIT profile)
MESG_NUM_FILE_ID = 0
MESG_NUM_DEVICE_INFO = 23
MESG_NUM_MONITORING = 55
MESG_NUM_MONITORING_INFO = 103
MESG_NUM_SLEEP_LEVEL = 275

# Garmin sleep_level enum (0 = unmeasurable)
SLEEP_LEVELS = {"AWAKE": 1, "LIGHT": 2, "DEEP": 3, "REM": 4}

# Field definitions (field_def_num, size_bytes, base_type)
# Base types: 0x84=uint16, 0x86=uint32, 0x02=uint8, 0x07=string, 0x8C=uint32z
ENUM   = 0x00  # enumeration (1 byte) - used for type fields like file_id.type
UINT8  = 0x02
UINT16 = 0x84
UINT32 = 0x86
UINT32Z = 0x8C


def to_garmin_ts(dt: datetime) -> int:
    """Convert a datetime to Garmin epoch seconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) - GARMIN_EPOCH


def fit_crc(data: bytes) -> int:
    """Compute the Garmin FIT CRC-16."""
    crc_table = [
        0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
    ]
    crc = 0
    for byte in data:
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc ^= tmp ^ crc_table[byte & 0xF]
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc ^= tmp ^ crc_table[(byte >> 4) & 0xF]
    return crc


# ---------------------------------------------------------------------------
# Low-level FIT record builders
# ---------------------------------------------------------------------------

def _definition_record(local_mesg_num: int, global_mesg_num: int, fields: list) -> bytes:
    """
    Build a Definition Message.
    fields: list of (field_def_num, size, base_type)
    """
    # Record header: 0x40 | local_mesg_num = definition message
    header = 0x40 | (local_mesg_num & 0x0F)
    reserved = 0x00
    architecture = 0x00  # little-endian
    num_fields = len(fields)

    field_bytes = b""
    for fdef, size, base_type in fields:
        field_bytes += struct.pack("BBB", fdef, size, base_type)

    body = struct.pack("<BBHB", reserved, architecture, global_mesg_num, num_fields)
    return struct.pack("B", header) + body + field_bytes


def _data_record(local_mesg_num: int, field_values: list, formats: str) -> bytes:
    """
    Build a Data Message.
    field_values: list of values matching the definition order
    formats: struct format string (little-endian assumed)
    """
    header = local_mesg_num & 0x0F
    body = struct.pack("<" + formats, *field_values)
    return struct.pack("B", header) + body


# ---------------------------------------------------------------------------
# High-level message builders
# ---------------------------------------------------------------------------

def _file_id_messages(manufacturer: int, product_id: int, serial_number: int, ts: int,
                      file_number: int = 100) -> bytes:
    """Emit the file_id definition + data record.
    Field order and field 6 (proprietary, 0xFFFF) match real Garmin firmware output.
    """
    defn = _definition_record(0, MESG_NUM_FILE_ID, [
        (3, 4, UINT32Z), # serial_number
        (4, 4, UINT32),  # time_created
        (1, 2, UINT16),  # manufacturer
        (2, 2, UINT16),  # product
        (5, 2, UINT16),  # number
        (6, 2, UINT16),  # proprietary field (always 0xFFFF in real firmware)
        (0, 1, ENUM),    # type (enum base type 0x00, NOT uint8)
    ])
    data = _data_record(0, [serial_number, ts, manufacturer, product_id, file_number, 0xFFFF, 32], "IIHHHHB")
    return defn + data


def _device_info_messages(manufacturer: int, product_id: int, serial_number: int, ts: int,
                          software_version: int = 331) -> bytes:
    """Emit device_info definition + data record (local mesg 1)."""
    defn = _definition_record(1, MESG_NUM_DEVICE_INFO, [
        (253, 4, UINT32),  # timestamp
        (2,   2, UINT16),  # manufacturer
        (4,   2, UINT16),  # product
        (3,   4, UINT32Z), # serial_number
        (5,   2, UINT16),  # software_version (raw: version * 100, e.g. 331 = 3.31)
    ])
    data = _data_record(1, [ts, manufacturer, product_id, serial_number, software_version], "IHHIH")
    return defn + data


def _monitoring_info_message(ts: int, utc_offset_seconds: int = 0) -> bytes:
    """Emit monitoring_info definition + data (local mesg 2).

    local_timestamp = Garmin UTC timestamp + local UTC offset in seconds.
    Garmin uses this to assign data to the correct local calendar day.
    """
    defn = _definition_record(2, MESG_NUM_MONITORING_INFO, [
        (253, 4, UINT32),  # timestamp
        (0,   4, UINT32),  # local_timestamp
        (1,   1, UINT8),   # activity_type
    ])
    local_ts = ts + utc_offset_seconds
    # Use activity_type=0 (generic) for the global file info.
    # Individual monitoring messages will specify walking (6) when steps are present.
    data = _data_record(2, [ts, local_ts, 0], "IIB")
    return defn + data


def _monitoring_messages(points: list[dict]) -> bytes:
    """
    Emit monitoring definitions, then an activity record per intraday point plus an HR
    record for each point that has a heart rate (None/0 means no reading: no HR record).
    points: [{"datetime": datetime, "heart_rate": int | None, "cumulative_steps": int, "steps_delta": int}, ...]

    Real Garmin devices emit separate monitoring records for activity/steps vs heart rate.
    When activity_type is present in a monitoring record, Garmin treats it as an activity
    summary and ignores heart_rate in the same record. HR must be in its own record.

    Local mesg 3: activity record — timestamp + cycles + activity_type (steps data)
    Local mesg 4: HR record      — timestamp + heart_rate (no activity_type)

    Cycles are cumulative WITHIN THIS SEGMENT ONLY, not from start of day.
    """
    defn_activity = _definition_record(3, MESG_NUM_MONITORING, [
        (253, 4, UINT32),  # timestamp
        (3,   4, UINT32),  # cycles (cumulative steps, within segment)
        (5,   1, UINT8),   # activity_type (6=walking makes cycles represent steps)
    ])
    defn_hr = _definition_record(4, MESG_NUM_MONITORING, [
        (253, 4, UINT32),  # timestamp
        (27,  1, UINT8),   # heart_rate
    ])
    records = defn_activity + defn_hr

    # Calculate segment-relative cumulative steps (reset at segment start)
    # The first point's own steps belong to this segment, so exclude them from the offset.
    segment_start_cumulative = (
        points[0].get("cumulative_steps", 0) - points[0].get("steps_delta", 0) if points else 0
    )

    for pt in points:
        ts = to_garmin_ts(pt["datetime"])
        hr = pt.get("heart_rate")  # None/0 means "no reading"; Garmin shows 0 bpm as no data
        # Segment-relative cumulative: steps accumulated within this segment only
        steps_cum_in_segment = pt.get("cumulative_steps", 0) - segment_start_cumulative
        cycles = max(0, steps_cum_in_segment)
        activity_type = 6 if pt.get("steps_delta", 0) > 0 else 0

        records += _data_record(3, [ts, cycles, activity_type], "IIB")
        if hr and hr > 0:
            # Offset HR record by 1 second to avoid Garmin aggregating cycles across records
            records += _data_record(4, [ts + 1, max(1, min(255, hr))], "IB")
    return records


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------

def _wrap_fit(messages: bytes) -> bytes:
    """Prepend the 14-byte FIT header and append the file CRC."""
    header = struct.pack(
        "<BBHI4sH",
        14,                    # header size
        FIT_PROTOCOL_VERSION,
        int(FIT_PROFILE_VERSION),
        len(messages),
        b".FIT",
        0x0000,                # header CRC placeholder (filled below)
    )
    header = header[:12] + struct.pack("<H", fit_crc(header[:12]))
    raw = header + messages
    return raw + struct.pack("<H", fit_crc(raw))


def build_monitoring_fit(
    points: list[dict],
    manufacturer: int,
    product_id: int,
    serial_number: int,
    software_version: int = 331,
    utc_offset_seconds: int = 0,
) -> bytes:
    """
    Build a complete, valid Garmin Monitoring FIT file (type 9).

    Args:
        points: merged intraday list from GoogleHealthClient.get_combined_intraday()
        manufacturer: Garmin manufacturer ID (usually 1)
        product_id: Garmin product ID from identity_grabber
        serial_number: Device serial number from identity_grabber
        utc_offset_seconds: local UTC offset in seconds (e.g. 3600 for UTC+1)

    Returns:
        Raw bytes of the .fit file ready to upload.
    """
    if not points:
        raise ValueError("No data points provided")

    first_ts = to_garmin_ts(points[0]["datetime"])
    # file_id.time_created is the start of the logged period, as on real devices.
    # Setting it to the upload time instead made Garmin keep only the last record(s)
    # of each segment (records older than time_created were dropped), leaving gaps.
    # Garmin deduplicates on (serial_number, time_created); a retry of the same
    # window then returns 409, which upload_fit treats as already uploaded.
    file_created_ts = first_ts

    # Build message payload
    messages = b""
    messages += _file_id_messages(manufacturer, product_id, serial_number, file_created_ts)
    messages += _device_info_messages(manufacturer, product_id, serial_number, first_ts, software_version)
    messages += _monitoring_info_message(first_ts, utc_offset_seconds)
    messages += _monitoring_messages(points)

    raw = _wrap_fit(messages)
    file_crc = int.from_bytes(raw[-2:], "little")

    log.info(
        "Built FIT file: %d points, %d bytes, CRC=0x%04X",
        len(points),
        len(raw),
        file_crc,
    )
    return raw


def _sleep_level_messages(stages: list[dict]) -> bytes:
    """
    Emit sleep_level definition + one record per minute of each sleep stage.

    stages: [{"start": datetime, "end": datetime, "type": "AWAKE|LIGHT|DEEP|REM"}, ...]
    Stages are non-overlapping; unknown types are skipped. Timestamps are ascending.
    """
    records = _definition_record(3, MESG_NUM_SLEEP_LEVEL, [
        (253, 4, UINT32),  # timestamp
        (0,   1, ENUM),    # sleep_level
    ])
    last_ts = -1
    for st in sorted(stages, key=lambda x: x["start"]):
        level = SLEEP_LEVELS.get(st["type"])
        if level is None:
            continue
        for ts in range(to_garmin_ts(st["start"]), to_garmin_ts(st["end"]), 60):
            if ts <= last_ts:
                continue  # never go backwards if stages overlap
            records += _data_record(3, [ts, level], "IB")
            last_ts = ts
    return records


def build_sleep_fit(
    stages: list[dict],
    manufacturer: int,
    product_id: int,
    serial_number: int,
    software_version: int = 331,
    utc_offset_seconds: int = 0,
) -> bytes:
    """
    Build a monitoring FIT file (type 32) carrying one night's sleep_level records.

    Experimental: whether Garmin Connect stores or displays this is unverified.
    """
    stages = [s for s in stages if s["type"] in SLEEP_LEVELS and s["end"] > s["start"]]
    if not stages:
        raise ValueError("No sleep stages provided")

    first_ts = to_garmin_ts(min(s["start"] for s in stages))
    messages = b""
    messages += _file_id_messages(manufacturer, product_id, serial_number, first_ts)
    messages += _device_info_messages(manufacturer, product_id, serial_number, first_ts, software_version)
    messages += _monitoring_info_message(first_ts, utc_offset_seconds)
    messages += _sleep_level_messages(stages)
    raw = _wrap_fit(messages)
    log.info("Built sleep FIT file: %d stage(s), %d bytes", len(stages), len(raw))
    return raw
