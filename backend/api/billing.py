"""backend/api/billing.py — Stripe billing SCAFFOLD (incomplete).

Status: scaffolding only. The endpoints are wired and safe to deploy — they
return HTTP 501 until Stripe is configured — but the actual checkout/activation
logic is left as TODOs (marked below) because it depends on product decisions
(pricing/plans) and a small schema addition.

`stripe` is imported lazily inside the handlers so the app and test suite run
without the package installed. To finish this:
  1. pip install stripe (add to backend/requirements.txt)
  2. Set STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_ID (see .env.example)
  3. Add a `stripe_customer_id` column to public.users so webhooks can map a
     Stripe customer back to a Courtvision user.
  4. Fill in the TODOs, then include this router in main.py.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth import get_user_id

router = APIRouter(prefix="/api/billing", tags=["billing"])

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")


class CheckoutRequest(BaseModel):
    # URLs Stripe redirects to after the hosted checkout completes/cancels.
    success_url: str
    cancel_url: str


@router.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest, user_id: str = Depends(get_user_id)):
    """Create a Stripe Checkout session for a coach to activate their team.

    SCAFFOLD: returns 501 until Stripe is configured. When configured, this
    should create a session for STRIPE_PRICE_ID tied to this user.
    """
    if not (STRIPE_SECRET_KEY and STRIPE_PRICE_ID):
        raise HTTPException(status_code=501, detail="Billing is not configured")

    import stripe  # lazy: only needed when billing is enabled
    stripe.api_key = STRIPE_SECRET_KEY

    # TODO: create/reuse a Stripe customer for `user_id` (store stripe_customer_id
    # on public.users), then create the Checkout session below. Passing
    # client_reference_id lets the webhook map the session back to the user.
    session = stripe.checkout.Session.create(
        mode="subscription",  # TODO: confirm subscription vs one-time per pricing model
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=req.success_url,
        cancel_url=req.cancel_url,
        client_reference_id=user_id,
    )
    return {"checkout_url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events (payment completed → activate the coach).

    SCAFFOLD: returns 501 until STRIPE_WEBHOOK_SECRET is set. When configured,
    verify the signature and, on `checkout.session.completed`, mark the coach
    (client_reference_id) as activated — mirroring api/activation.py's
    activated_at update.
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=501, detail="Billing is not configured")

    import stripe  # lazy

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        # Signature/parse failure — do not leak details.
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        user_id = event["data"]["object"].get("client_reference_id")
        # TODO: set users.activated_at = now() for `user_id` (and assign/record
        # the activation_key), so the coach's team is unlocked. Use the
        # service-role Supabase client as api/activation.py does. Guard against
        # duplicate webhook deliveries (idempotency).
        _ = user_id

    return {"received": True}
