"""Guards that the billing SCAFFOLD is safe to ship (P4.2 scaffold).

The billing router must import without the `stripe` package and stay inert until
configured — so it can be deployed now without breaking anything. Run with:
pytest tests/test_billing_scaffold.py
"""

from fastapi.testclient import TestClient

import main  # noqa: E402


def test_billing_router_mounted():
    paths = {r.path for r in main.app.routes}
    assert "/api/billing/webhook" in paths
    assert "/api/billing/create-checkout-session" in paths


def test_webhook_returns_501_when_unconfigured():
    # No STRIPE_WEBHOOK_SECRET in the test env → the scaffold is inert, not broken.
    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.post("/api/billing/webhook", content=b"{}")
    assert resp.status_code == 501
