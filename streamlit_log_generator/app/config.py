# app/config.py

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    ROLE_WEIGHTS_NORMAL: List[float] = field(default_factory=lambda: [0.15, 0.40, 0.35, 0.10])  # ["Doctor", "Nurse", "Staff", "Admin"]
    NORMAL_METHODS: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    NORMAL_METHOD_WEIGHTS: List[float] = field(default_factory=lambda: [0.65, 0.20, 0.05, 0.02, 0.03, 0.03, 0.02])
    HTTP_RESPONSE_CODES_NORMAL: Dict[str, List] = field(default_factory=lambda: {
        "codes": [200, 201, 302, 304],
        "weights": [0.70, 0.15, 0.10, 0.05]
    })
    HTTP_RESPONSE_CODES_ANOMALOUS: Dict[str, List] = field(default_factory=lambda: {
        "codes": [200, 403, 404, 500],
        "weights": [0.2, 0.4, 0.2, 0.2]
    })
    ROLES: Dict[str, List[int]] = field(default_factory=lambda: {
        "Doctor": list(range(1, 21)),    # 20 Doctors
        "Nurse": list(range(21, 46)),    # 25 Nurses
        "Staff": list(range(46, 61)),    # 15 Staff
        "Admin": list(range(61, 71)),    # 10 Admin
    })
    ENDPOINTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Doctor": [
            "/patient/records",
            "/patient/labs",
            "/patient/appointments",
            "/billing/invoices"
        ],
        "Nurse": [
            "/patient/records",
            "/patient/labs",
            "/patient/appointments",
            "/billing/invoices"
        ],
        "Staff": [
            "/inventory/items",
            "/billing/invoices",
            "/patient/scheduling"
        ],
        "Admin": [
            "/admin/settings",
            "/admin/credentials"
        ],
    })
    MASTER_LOG_FILE: str = "app/data/master_logs.csv"
    INFERENCE_LOG_FILE: str = "app/data/inference_logs.csv"
    NETWORK_SUBNET: str = "10.0.0"
    ANOMALY_RATIO: float = 0.02
    ROLE_SWITCH_PROBABILITY: float = 0.05
    ANOMALY_IP_PROBABILITY: float = 0.10
    AFTER_HOURS_PROBABILITY: float = 0.25
    EXTENDED_DAYS: int = 90
    TOTAL_LOGS_DEFAULT: int = 100000
