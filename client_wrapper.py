import requests
import hashlib
from crypto_provider import generate_exchange_keypair, kem_decapsulate

class QSRACClient:
    def __init__(self, base_url):
        self.base_url = base_url

        self.priv, self.pub = generate_exchange_keypair()

        self.session_id = None
        self.session_key = None
        self.seq = 0
        self.envelope = hashlib.sha256(b"init").hexdigest()
        self.core_token = None

    def login(self, username):
        r = requests.post(f"{self.base_url}/login", json={
            "username": username,
            "client_public_key": self.pub.hex()
        }).json()

        self.session_id = r["session_id"]
        self.core_token = r["core_token"]

        ciphertext = bytes.fromhex(r["kem_ciphertext"])
        self.session_key = kem_decapsulate(self.priv, ciphertext)

        self.seq = r["seq"]
        self.envelope = r["init_envelope"]

    def request(self, path, context):
        headers = {
            "X-Session-ID": self.session_id,
            "X-Core-Token": self.core_token,
            "X-QSRAC-Seq": str(self.seq),
            "X-QSRAC-Envelope": self.envelope,
        }

        for k, v in context.items():
            headers[f"X-{k.replace('_','-').title()}"] = str(v)

        r = requests.get(f"{self.base_url}{path}", headers=headers)

        if r.status_code == 200:
            self.seq = int(r.headers["X-QSRAC-Seq"])
            self.envelope = r.headers["X-QSRAC-Envelope"]

        return r