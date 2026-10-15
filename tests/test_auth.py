import pyotp
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pi_p.app import server
from pi_p.auth import COOKIE, MAX_ATTEMPTS, Auth

SECRET = pyotp.random_base32()


def test_code_cannot_be_reused():
    auth = Auth(SECRET)
    code = pyotp.TOTP(SECRET).now()
    assert auth.verify(code)
    assert not auth.verify(code)


def test_tampered_or_expired_token_is_rejected():
    auth = Auth(SECRET)
    token = auth.issue()
    assert auth.valid(token)
    expires, sig = token.split(".")
    assert not auth.valid(f"{int(expires) + 999}.{sig}")
    assert not auth.valid(f"1.{auth._sign('1')}")
    assert not Auth(pyotp.random_base32()).valid(token)


def test_disabled_auth_allows_everything():
    assert Auth(None).valid(None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "auth", Auth(SECRET))
    return TestClient(server.app)


def test_api_requires_login(client):
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/session").json() == {"enabled": True, "authenticated": False}
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()
    assert e.value.code == 4401


def test_login_flow(client):
    assert client.post("/api/login", json={"code": "000000"}).status_code == 401
    r = client.post("/api/login", json={"code": pyotp.TOTP(SECRET).now()})
    assert r.status_code == 200
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert client.get("/api/models").status_code == 200
    client.post("/api/logout")
    client.cookies.clear()
    assert client.get("/api/models").status_code == 401


def test_login_is_rate_limited(client):
    for _ in range(MAX_ATTEMPTS):
        client.post("/api/login", json={"code": "000000"})
    assert client.post("/api/login", json={"code": pyotp.TOTP(SECRET).now()}).status_code == 429
    assert client.cookies.get(COOKIE) is None
