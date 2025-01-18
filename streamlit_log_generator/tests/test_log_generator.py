# tests/test_log_generator.py

import unittest
from app.log_generator import generate_normal_log, generate_anomalous_log, generate_logs
from app.config import Config

class TestLogGenerator(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_generate_normal_log(self):
        log = generate_normal_log(
            roles=self.config.ROLES,
            endpoints=self.config.ENDPOINTS,
            network_subnet=self.config.NETWORK_SUBNET,
            extended_days=self.config.EXTENDED_DAYS
        )
        self.assertIn("LogID", log)
        self.assertIn("UserID", log)
        self.assertEqual(log["Anomalous"], 0)

    def test_generate_anomalous_log(self):
        logs = generate_anomalous_log(
            roles=self.config.ROLES,
            endpoints=self.config.ENDPOINTS,
            network_subnet=self.config.NETWORK_SUBNET,
            extended_days=self.config.EXTENDED_DAYS
        )
        self.assertTrue(len(logs) >= 1)
        for log in logs:
            self.assertIn("LogID", log)
            self.assertIn("UserID", log)
            self.assertEqual(log["Anomalous"], 1)

    def test_generate_logs(self):
        total_logs = 100
        anomaly_ratio = 0.05
        logs = generate_logs(
            total_logs=total_logs,
            anomaly_ratio=anomaly_ratio,
            roles=self.config.ROLES,
            endpoints=self.config.ENDPOINTS,
            network_subnet=self.config.NETWORK_SUBNET,
            extended_days=self.config.EXTENDED_DAYS
        )
        self.assertEqual(len(logs), total_logs)
        anomalies = sum(log["Anomalous"] for log in logs)
        self.assertAlmostEqual(anomalies / total_logs, anomaly_ratio, delta=0.01)

if __name__ == "__main__":
    unittest.main()
