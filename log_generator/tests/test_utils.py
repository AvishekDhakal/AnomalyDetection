# tests/test_utils.py

import unittest
from utils import build_parameters

class TestUtils(unittest.TestCase):
    def test_build_parameters_with_id_range(self):
        config = {
            "PARAMETERS": {
                "/patient/records": {
                    "id_range": [1000, 9999],
                    "export": True,
                    "limit": 1000
                }
            }
        }
        endpoint = "/patient/records"
        param_str = build_parameters(endpoint, config)
        self.assertTrue(param_str.startswith("/"))
        self.assertIn("export=true", param_str)
        self.assertIn("limit=1000", param_str)
    
    def test_build_parameters_without_id_range(self):
        config = {
            "PARAMETERS": {
                "/patient/appointments/confirm": {}
            }
        }
        endpoint = "/patient/appointments/confirm"
        param_str = build_parameters(endpoint, config)
        self.assertEqual(param_str, "")
    
    def test_build_parameters_missing_parameters(self):
        config = {
            "PARAMETERS": {}
        }
        endpoint = "/unknown/endpoint"
        param_str = build_parameters(endpoint, config)
        self.assertEqual(param_str, "")
    
    def test_build_parameters_with_anomaly_payload(self):
        config = {
            "PARAMETERS": {
                "/patient/records": {
                    "id_range": [1000, 9999],
                    "export": True,
                    "limit": 1000
                }
            }
        }
        endpoint = "/patient/records"
        anomaly_payload = "; DROP TABLE users"
        param_str = build_parameters(endpoint, config, is_anomalous=True, anomaly_payload=anomaly_payload)
        self.assertIn(anomaly_payload, param_str)

if __name__ == '__main__':
    unittest.main()
