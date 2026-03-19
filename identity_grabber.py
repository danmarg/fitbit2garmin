"""
identity_grabber.py
Utility to extract manufacturer, product_id, and serial_number from a real
Garmin .fit file so Fitbit2Garmin can impersonate the same device.

Usage:
    python identity_grabber.py path/to/activity.fit
"""

import sys
import json
import fitparse


def extract_device_identity(fit_path: str) -> dict:
    fitfile = fitparse.FitFile(fit_path)
    identity = {}

    for record in fitfile.get_messages("file_id"):
        for field in record:
            if field.name == "manufacturer":
                identity["manufacturer"] = field.raw_value
            elif field.name in ("product", "garmin_product"):
                identity["product_id"] = field.raw_value
            elif field.name == "serial_number":
                identity["serial_number"] = field.raw_value
            elif field.name == "type":
                identity["file_type"] = str(field.value)

    for record in fitfile.get_messages("device_info"):
        for field in record:
            if field.name == "device_index" and field.raw_value == 0:
                # Primary device row
                pass
            if field.name == "software_version":
                identity["software_version"] = field.value

    return identity


def main():
    if len(sys.argv) < 2:
        print("Usage: python identity_grabber.py <path_to_fit_file>")
        sys.exit(1)

    fit_path = sys.argv[1]
    print(f"Parsing: {fit_path}\n")

    try:
        identity = extract_device_identity(fit_path)
    except Exception as e:
        print(f"Error parsing FIT file: {e}")
        sys.exit(1)

    if not identity:
        print("No file_id message found in this FIT file.")
        sys.exit(1)

    print("Extracted device identity:")
    print(json.dumps(identity, indent=2))
    print()
    print("Add to config.yaml:")
    print("device:")
    print(f"  manufacturer: {identity.get('manufacturer', 'UNKNOWN')}")
    print(f"  product_id:   {identity.get('product_id', 'UNKNOWN')}")
    print(f"  serial_number: {identity.get('serial_number', 'UNKNOWN')}")


if __name__ == "__main__":
    main()
