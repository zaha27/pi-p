import argparse
import hashlib
import hmac
import time
from collections import deque
from pathlib import Path

import pyotp
import qrcode

COOKIE = "pi_p_session"
SESSION_SECONDS = 12 * 3600
MAX_ATTEMPTS = 5
ATTEMPT_WINDOW = 60
STEP = 30


class Auth:
    def __init__(self, secret: str | None):
        self.totp = pyotp.TOTP(secret) if secret else None
        self.key = hashlib.sha256(b"pi-p-session:" + secret.encode()).digest() if secret else b""
        self.attempts: dict[str, deque[float]] = {}
        self.last_step = -1

    @property
    def enabled(self) -> bool:
        return self.totp is not None

    def allow_attempt(self, client: str) -> bool:
        now = time.time()
        recent = self.attempts.setdefault(client, deque())
        while recent and now - recent[0] > ATTEMPT_WINDOW:
            recent.popleft()
        if len(recent) >= MAX_ATTEMPTS:
            return False
        recent.append(now)
        return True

    def verify(self, code: str) -> bool:
        if not self.totp or len(code) != 6 or not code.isdigit():
            return False
        current = int(time.time() // STEP)
        for step in (current - 1, current, current + 1):
            if step > self.last_step and hmac.compare_digest(self.totp.at(step * STEP), code):
                self.last_step = step
                return True
        return False

    def _sign(self, expires: str) -> str:
        return hmac.new(self.key, expires.encode(), "sha256").hexdigest()

    def issue(self) -> str:
        expires = str(int(time.time()) + SESSION_SECONDS)
        return f"{expires}.{self._sign(expires)}"

    def valid(self, token: str | None) -> bool:
        if not self.enabled:
            return True
        expires, _, signature = (token or "").partition(".")
        return expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(signature, self._sign(expires))


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    if not path.is_file():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def save_secret(secret: str, path: Path = Path(".env")) -> None:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("PI_P_TOTP_SECRET=")] if path.is_file() else []
    path.write_text("\n".join([*lines, f"PI_P_TOTP_SECRET={secret}"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="înlocuiește secretul existent")
    args = ap.parse_args()
    if load_env().get("PI_P_TOTP_SECRET") and not args.force:
        raise SystemExit("Există deja un secret în .env; folosește --force ca să-l înlocuiești (codurile vechi nu vor mai merge)")
    secret = pyotp.random_base32()
    uri = pyotp.TOTP(secret).provisioning_uri(name="pi-p", issuer_name="Proiect Prelucrarea Imaginilor")
    qr = qrcode.QRCode(border=1)
    qr.add_data(uri)
    qr.print_ascii(invert=True)
    save_secret(secret)
    print(f"\nScanează codul cu Google Authenticator (sau introdu manual cheia {secret}).")
    print("Secretul a fost salvat în .env. Repornește serverul ca să-l folosească.")


if __name__ == "__main__":
    main()
