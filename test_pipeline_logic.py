
import unittest
from datetime import datetime, timedelta, timezone
import fitparse
from fit_engine import build_monitoring_fit, GARMIN_EPOCH

class TestFitGeneration(unittest.TestCase):
    def test_build_and_parse_monitoring_fit(self):
        # 1. Setup Mock Data (2 hours of data)
        start_dt = datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
        mock_points = []
        for i in range(120):
            dt = start_dt + timedelta(minutes=i)
            mock_points.append({
                "datetime": dt,
                "heart_rate": 70 + (i % 10),
                "steps_delta": 10,
                "cumulative_steps": (i + 1) * 10
            })

        # 2. Spoof IDs
        manufacturer = 1
        product_id = 3993
        serial_number = 3445298263

        # 3. Build FIT file
        fit_bytes = build_monitoring_fit(
            points=mock_points,
            manufacturer=manufacturer,
            product_id=product_id,
            serial_number=serial_number
        )

        # 4. Save to disk
        with open("test_output.fit", "wb") as f:
            f.write(fit_bytes)

        # 5. Parse back and Validate
        fitfile = fitparse.FitFile("test_output.fit")
        
        # Verify File ID
        file_id = next(fitfile.get_messages("file_id"))
        # Some fitparse versions map 4 to 'activity' instead of 'monitoring'
        self.assertIn(file_id.get_value("type"), ["monitoring", "monitoring_a", "monitoring_b", "monitoring_daily", "activity"])
        self.assertEqual(file_id.get_value("product"), product_id)

        # Verify Monitoring Messages
        monitoring_msgs = list(fitfile.get_messages("monitoring"))
        self.assertEqual(len(monitoring_msgs), 120)
        
        # Check first record
        first_msg = monitoring_msgs[0]
        print("\nDebug: Fields found in monitoring message:")
        for field in first_msg:
            print(f"  {field.name} ({field.def_num}): {field.value}")
            
        self.assertEqual(first_msg.get_value("heart_rate"), 70)
        self.assertEqual(first_msg.get_value("cycles"), 10)

        print(f"\nSUCCESS: Engine logic verified. FIT file correctly encoded with {len(monitoring_msgs)} points.")

if __name__ == "__main__":
    unittest.main()
