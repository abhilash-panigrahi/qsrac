from client_wrapper import QSRACClient

BASE = "http://localhost:8000"

client = QSRACClient(BASE)

client.login("user")

res = client.request("/test", {
    "hour_of_day": 10,
    "request_rate": 1.0,
    "failed_attempts": 0,
    "geo_risk_score": 0.1,
    "device_trust_score": 0.9,
    "sensitivity_level": 1,
    "is_vpn": 0,
    "is_tor": 0
})

print(res.status_code, res.json())
print("SEQ:", client.seq)