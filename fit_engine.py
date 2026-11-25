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

# Field definitions (field_def_num, size_bytes, base_type)
# Base types: 0x84=uint16, 0x86=uint32, 0x02=uint8, 0x07=string, 0x8C=uint32z
ENUM   = 0x00  # enumeration (1 byte) - used for type fields like file_id.type
UINT8  = 0x02
UINT16 = 0x84
UINT32 = 0x86
UINT32Z = 0x8C
SINT16 = 0x83


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

    Garmin discards a monitoring file whose cycles never rise above zero ("File was a
    valid type, but contained no content"), heart rate included. A segment with no steps
    therefore gets one walking cycle on its first record; Garmin did not show it as a step
    when tested (2026-11-22).
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

    # Cycles are summed from each minute's steps within this segment, so they never go
    # backwards (cumulative_steps resets at local midnight, which a segment may cross).
    no_steps = all(pt.get("steps_delta", 0) <= 0 for pt in points)
    steps_in_segment = 0

    for i, pt in enumerate(points):
        ts = to_garmin_ts(pt["datetime"])
        hr = pt.get("heart_rate")  # None/0 means "no reading"; Garmin shows 0 bpm as no data
        steps_in_segment += max(0, pt.get("steps_delta", 0))
        cycles = steps_in_segment
        activity_type = 6 if pt.get("steps_delta", 0) > 0 else 0
        if no_steps:
            cycles = 1
            activity_type = 6 if i == 0 else 0

        records += _data_record(3, [ts, cycles, activity_type], "IIB")
        if hr and hr > 0:
            # Offset HR record by 1 second to avoid Garmin aggregating cycles across records
            records += _data_record(4, [ts + 1, max(1, min(255, hr))], "IB")
    return records


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------

def build_monitoring_fit(
    points: list[dict],
    manufacturer: int,
    product_id: int,
    serial_number: int,
    software_version: int = 331,
    utc_offset_seconds: int = 0,
    time_created_offset: int = 0,
) -> bytes:
    """
    Build a complete, valid Garmin Monitoring FIT file (file_id.type 32, monitoring_b).

    time_created_offset moves file_id.time_created that many seconds before the first
    point. Garmin identifies a file by time_created, so a re-upload of the same span
    needs a different value or it is rejected as a duplicate.

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
    file_created_ts = first_ts - time_created_offset

    # Build message payload
    messages = b""
    messages += _file_id_messages(manufacturer, product_id, serial_number, file_created_ts)
    messages += _device_info_messages(manufacturer, product_id, serial_number, first_ts, software_version)
    messages += _monitoring_info_message(first_ts, utc_offset_seconds)
    messages += _monitoring_messages(points)

    # FIT file header (14 bytes)
    data_size = len(messages)
    header = struct.pack(
        "<BBHI4sH",
        14,                    # header size
        FIT_PROTOCOL_VERSION,
        int(FIT_PROFILE_VERSION),
        data_size,
        b".FIT",
        0x0000,                # header CRC placeholder (filled below)
    )
    header_crc = fit_crc(header[:12])
    header = header[:12] + struct.pack("<H", header_crc)

    raw = header + messages
    file_crc = fit_crc(raw)
    raw += struct.pack("<H", file_crc)

    log.info(
        "Built FIT file: %d points, %d bytes, CRC=0x%04X",
        len(points),
        len(raw),
        file_crc,
    )
    return raw


# ---------------------------------------------------------------------------
# Sleep
# ---------------------------------------------------------------------------

# time_created must differ between files with the same serial, or Garmin rejects the
# later one as a duplicate. Every file starts its time_created at a fixed second of the
# minute so the kinds can never meet:
#   heart rate :00 (first point)        backfill :45–:59 (main.BACKFILL_MAX_SHIFT)
#   SpO2/breathing :19 (build_vitals_fit) HRV :29, sleep :30 (from the last stage end,
#   floored to the minute)
SLEEP_CREATED_DELAY = 90

MESG_NUM_FILE_CREATOR = 49
MESG_NUM_SLEEP_LEVEL = 275
MESG_NUM_SLEEP_ASSESSMENT = 346
FILE_TYPE_SLEEP = 49
SLEEP_LEVELS = {"AWAKE": 1, "LIGHT": 2, "DEEP": 3, "REM": 4}
MESG_NUM_EVENT = 21
MESG_NUM_SLEEP_DATA_INFO = 273
EVENT_SLEEP = 74


def _sleep_level_records(stages: list[dict]) -> tuple[int, int, list[tuple[int, int]]]:
    """
    Turn stages into (start_ts, end_ts, [(ts, level), ...]) in Garmin epoch seconds.

    A real watch writes one sleep_level record per stage, stamped with the stage's END
    (verified 2026-11-23: Garmin showed light 22:52:59–23:01:59 for a record
    "23:01:59 level 2" after the 22:52:59 sleep-start event). Overlaps are clipped,
    gaps between stages become AWAKE, and adjacent equal levels are merged.
    """
    stages = sorted(
        (s for s in stages if s["type"] in SLEEP_LEVELS and s["end"] > s["start"]),
        key=lambda x: x["start"],
    )
    if not stages:
        raise ValueError("No sleep stages provided")
    start_ts = cursor = to_garmin_ts(stages[0]["start"])
    out: list[tuple[int, int]] = []

    def emit(ts, level):
        if out and out[-1][1] == level:
            out[-1] = (ts, level)
        else:
            out.append((ts, level))

    for st in stages:
        s, e = to_garmin_ts(st["start"]), to_garmin_ts(st["end"])
        if e <= cursor:
            continue
        if s > cursor:
            emit(s, SLEEP_LEVELS["AWAKE"])
        emit(e, SLEEP_LEVELS[st["type"]])
        cursor = e
    return start_ts, cursor, out


# sleep_assessment (346) fields, in the order the FR255S writes them; names follow the
# FIT SDK. Fields not given are written as invalid. Garmin shows overall_sleep_score as
# the night's score, and stores nothing without real values here (2026-11-23).
SLEEP_ASSESSMENT_FIELDS = [
    # (field_def_num, name, size, base_type, struct format)
    (15, "average_stress_during_sleep", 2, UINT16, "H"),  # stress x 100
    (0,  "combined_awake_score", 1, UINT8, "B"),
    (1,  "awake_time_score", 1, UINT8, "B"),
    (2,  "awakenings_count_score", 1, UINT8, "B"),
    (3,  "deep_sleep_score", 1, UINT8, "B"),
    (4,  "sleep_duration_score", 1, UINT8, "B"),
    (5,  "light_sleep_score", 1, UINT8, "B"),
    (6,  "overall_sleep_score", 1, UINT8, "B"),
    (7,  "sleep_quality_score", 1, UINT8, "B"),
    (8,  "sleep_recovery_score", 1, UINT8, "B"),
    (9,  "rem_sleep_score", 1, UINT8, "B"),
    (10, "sleep_restlessness_score", 1, UINT8, "B"),
    (11, "awakenings_count", 1, UINT8, "B"),
    (12, "unknown_12", 1, ENUM, "B"),
    (13, "unknown_13", 1, ENUM, "B"),
    (14, "interruptions_score", 1, UINT8, "B"),
    (16, "unknown_16", 1, ENUM, "B"),
]


def _sleep_assessment_message(local_mesg_num: int, values: dict) -> bytes:
    defn = _definition_record(local_mesg_num, MESG_NUM_SLEEP_ASSESSMENT,
                              [(num, size, base) for num, _, size, base, _ in SLEEP_ASSESSMENT_FIELDS])
    invalid = {"H": 0xFFFF, "B": 0xFF}
    row = [values.get(name, invalid[fmt]) for _, name, _, _, fmt in SLEEP_ASSESSMENT_FIELDS]
    fmt = "".join(f for *_, f in SLEEP_ASSESSMENT_FIELDS)
    return defn + _data_record(local_mesg_num, row, fmt)


def _wrap_fit(messages: bytes) -> bytes:
    """Prepend the 14-byte FIT header and append the file CRC."""
    header = struct.pack("<BBHI4sH", 14, FIT_PROTOCOL_VERSION, int(FIT_PROFILE_VERSION),
                         len(messages), b".FIT", 0x0000)
    header = header[:12] + struct.pack("<H", fit_crc(header[:12]))
    raw = header + messages
    return raw + struct.pack("<H", fit_crc(raw))


def build_sleep_fit(
    stages: list[dict],
    manufacturer: int,
    product_id: int,
    serial_number: int,
    assessment: dict,
    software_version: int = 331,
    utc_offset_seconds: int = 0,
    created_delay: int = SLEEP_CREATED_DELAY,
) -> bytes:
    """
    Build a sleep FIT file (file_id.type 49) for one night, laid out like the file a
    Forerunner 255S writes (GBN71003.fit, which Garmin accepted on 2026-11-23):
    file_id, file_creator, device_info, sleep_data_info (273), sleep start/stop events
    (74), sleep_assessment (346), then sleep_level (275) records stamped at each
    stage's END. `assessment` maps SLEEP_ASSESSMENT_FIELDS names to values; Garmin
    ignores the night without it. time_created is created_delay seconds after waking.
    """
    start_ts, end_ts, levels = _sleep_level_records(stages)
    created = end_ts // 60 * 60 + created_delay  # whole minute first: see SLEEP_CREATED_DELAY

    m = b""
    m += _definition_record(0, MESG_NUM_FILE_ID, [
        (3, 4, UINT32Z),  # serial_number
        (4, 4, UINT32),   # time_created
        (7, 4, UINT32),   # unknown (invalid on the watch)
        (1, 2, UINT16),   # manufacturer
        (2, 2, UINT16),   # product
        (5, 2, UINT16),   # number (invalid on the watch)
        (0, 1, ENUM),     # type
    ])
    m += _data_record(0, [serial_number, created, 0xFFFFFFFF, manufacturer, product_id, 0xFFFF, FILE_TYPE_SLEEP],
                      "IIIHHHB")
    m += _definition_record(1, MESG_NUM_FILE_CREATOR, [(0, 2, UINT16)])  # software_version
    m += _data_record(1, [software_version], "H")
    m += _device_info_messages(manufacturer, product_id, serial_number, created, software_version)
    m += _definition_record(3, MESG_NUM_SLEEP_DATA_INFO, [
        (253, 4, UINT32),  # timestamp
        (2,   4, UINT32),  # local timestamp
        (1,   2, UINT16),  # sample interval (s), 60 on the watch
        (0,   1, UINT8),   # 2 on the watch
        (3,   1, ENUM),    # 1 on the watch
    ])
    m += _data_record(3, [created, created + utc_offset_seconds, 60, 2, 1], "IIHBB")
    m += _definition_record(4, MESG_NUM_EVENT, [
        (253, 4, UINT32),  # timestamp
        (3,   4, UINT32),  # data: local timestamp
        (0,   1, ENUM),    # event (74 = sleep)
        (1,   1, ENUM),    # event_type (0 start, 1 stop)
    ])
    m += _data_record(4, [start_ts, start_ts + utc_offset_seconds, EVENT_SLEEP, 0], "IIBB")
    m += _data_record(4, [end_ts, end_ts + utc_offset_seconds, EVENT_SLEEP, 1], "IIBB")
    m += _sleep_assessment_message(6, assessment)
    m += _definition_record(5, MESG_NUM_SLEEP_LEVEL, [
        (253, 4, UINT32),  # timestamp (end of the stage)
        (0,   1, ENUM),    # sleep_level
    ])
    for ts, level in levels:
        m += _data_record(5, [ts, level], "IB")

    raw = _wrap_fit(m)
    log.info("Built sleep FIT file: %d level record(s), %d bytes", len(levels), len(raw))
    return raw


# ---------------------------------------------------------------------------
# Overnight vitals: SpO2 and respiration (monitoring_b), HRV (hrv status)
# ---------------------------------------------------------------------------

FILE_TYPE_HRV_STATUS = 68
MESG_NUM_SPO2_DATA = 269
MESG_NUM_RESPIRATION_RATE = 297
MESG_NUM_HRV_STATUS_SUMMARY = 370
MESG_NUM_HRV_VALUE = 371
SPO2_MODE_PERIODIC = 3


def build_vitals_fit(
    spo2: list[tuple[datetime, float]],
    respiration: list[tuple[datetime, float]],
    manufacturer: int,
    product_id: int,
    serial_number: int,
    software_version: int = 331,
    utc_offset_seconds: int = 0,
) -> bytes:
    """
    Build a monitoring_b file (type 32) with SpO2 (269, %) and respiration (297,
    breaths/min) readings, as the FR255S writes them overnight (MBP75600.FIT). Garmin
    accepted this layout on 2026-11-25 and showed both on the night's sleep page, even
    though the sleep file had gone up earlier. The readings must lie within one local
    date (Garmin discards monitoring files that span local midnight).

    The one walking cycle keeps Garmin from discarding the file as empty (see
    _monitoring_messages). time_created is 41 s before the first whole minute: records
    before time_created are dropped, and Garmin treats a second file with the same
    serial and time_created as a duplicate, so it must miss the other files' seconds
    (see SLEEP_CREATED_DELAY). If that would fall before local midnight, it is 19 s
    after midnight instead. Raises ValueError if no reading is left after that.
    """
    times = [t for t, _ in spo2] + [t for t, _ in respiration]
    if not times:
        raise ValueError("No SpO2 or respiration readings provided")
    created = to_garmin_ts(min(times).replace(second=0, microsecond=0)) - 41
    # ...but never before local midnight: a time_created on the previous local date would
    # make the file span midnight. Readings before it (at most the first minute) are dropped.
    local_midnight = (created + utc_offset_seconds) // 86400 * 86400 - utc_offset_seconds
    if local_midnight != (to_garmin_ts(max(times)) + utc_offset_seconds) // 86400 * 86400 - utc_offset_seconds:
        created = local_midnight + 86400 + 19
    spo2 = [(t, v) for t, v in spo2 if to_garmin_ts(t) >= created]
    respiration = [(t, v) for t, v in respiration if to_garmin_ts(t) >= created]
    if not spo2 and not respiration:
        raise ValueError("No SpO2 or respiration readings after local midnight")

    m = b""
    m += _file_id_messages(manufacturer, product_id, serial_number, created, file_number=0xFFFF)  # type 32
    m += _device_info_messages(manufacturer, product_id, serial_number, created, software_version)
    m += _definition_record(2, MESG_NUM_MONITORING_INFO, [(253, 4, UINT32), (0, 4, UINT32), (1, 1, ENUM)])
    m += _data_record(2, [created, created + utc_offset_seconds, 6], "IIB")
    m += _definition_record(3, MESG_NUM_MONITORING, [(253, 4, UINT32), (3, 4, UINT32), (5, 1, ENUM)])
    m += _data_record(3, [created, 1, 6], "IIB")
    m += _definition_record(4, MESG_NUM_RESPIRATION_RATE, [(253, 4, UINT32), (0, 2, SINT16)])
    m += _definition_record(5, MESG_NUM_SPO2_DATA, [(253, 4, UINT32), (0, 1, UINT8), (1, 1, UINT8), (2, 1, ENUM)])
    records = sorted([(to_garmin_ts(t), 4, v) for t, v in respiration] + [(to_garmin_ts(t), 5, v) for t, v in spo2])
    for ts, local, v in records:
        if local == 4:
            m += _data_record(4, [ts, round(v * 100)], "Ih")
        else:  # confidence unknown: invalid
            m += _data_record(5, [ts, max(0, min(100, round(v))), 0xFF, SPO2_MODE_PERIODIC], "IBBB")
    raw = _wrap_fit(m)
    log.info("Built vitals FIT file: %d SpO2, %d respiration reading(s), %d bytes",
             len(spo2), len(respiration), len(raw))
    return raw


def _ms128(ms: float) -> int:
    """HRV in ms as uint16 / 128, clamped below the invalid value."""
    return max(0, min(0xFFFE, round(ms * 128)))


def build_hrv_fit(
    values: list[tuple[datetime, float]],
    created: datetime,
    manufacturer: int,
    product_id: int,
    serial_number: int,
    software_version: int = 331,
) -> bytes:
    """
    Build an HRV status file (type 68) like the FR255S's HRVStatus/GBP74437.fit: 5-minute
    RMSSD values (371, ms) and a summary (370) with the night's average and 5-minute high.
    The baseline and weekly average are left invalid and the status "none": they come
    from the user's history, which only Garmin has. The watch stamps this file 1 s before
    the sleep file.
    """
    if not values:
        raise ValueError("No HRV values provided")
    ts = to_garmin_ts(created)
    ms = [v for _, v in values]
    m = b""
    m += _definition_record(0, MESG_NUM_FILE_ID, [
        (3, 4, UINT32Z), (4, 4, UINT32), (7, 4, UINT32), (1, 2, UINT16), (2, 2, UINT16), (5, 2, UINT16), (0, 1, ENUM),
    ])
    m += _data_record(0, [serial_number, ts, 0xFFFFFFFF, manufacturer, product_id, 0xFFFF, FILE_TYPE_HRV_STATUS],
                      "IIIHHHB")
    m += _definition_record(1, MESG_NUM_FILE_CREATOR, [(0, 2, UINT16)])
    m += _data_record(1, [software_version], "H")
    m += _device_info_messages(manufacturer, product_id, serial_number, to_garmin_ts(values[0][0]), software_version)
    m += _definition_record(3, MESG_NUM_HRV_VALUE, [(253, 4, UINT32), (0, 2, UINT16)])
    for t, v in values:
        m += _data_record(3, [to_garmin_ts(t), _ms128(v)], "IH")
    m += _definition_record(4, MESG_NUM_HRV_STATUS_SUMMARY, [
        (253, 4, UINT32), (0, 2, UINT16), (1, 2, UINT16), (2, 2, UINT16), (3, 2, UINT16), (4, 2, UINT16),
        (5, 2, UINT16), (6, 1, ENUM),
    ])
    m += _data_record(4, [ts, 0xFFFF, _ms128(sum(ms) / len(ms)), _ms128(max(ms)),
                          0xFFFF, 0xFFFF, 0xFFFF, 0], "IHHHHHHB")
    raw = _wrap_fit(m)
    log.info("Built HRV FIT file: %d value(s), %d bytes", len(values), len(raw))
    return raw
