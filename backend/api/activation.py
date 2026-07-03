from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import os
from supabase import create_client, Client
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id
from rate_limit import limiter

router = APIRouter(prefix="/api/activation", tags=["activation"])

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


class ActivationRequest(BaseModel):
    activation_key: str


@router.post("/activate")
@limiter.limit("10/15minutes")
async def activate_key(request: Request, activation_data: ActivationRequest, user_id: str = Depends(get_user_id)):
    """
    Activate a coach account with an activation key.
    Only coaches can activate accounts.
    Activation keys are manually added to the database by admin after payment.
    """
    # Verify user is a coach
    user_response = supabase.table("users").select("role, activated_at").eq("id", user_id).single().execute()
    
    if not user_response.data or user_response.data.get("role") != "coach":
        raise HTTPException(status_code=403, detail="Only coaches can activate accounts")
    
    # Check if user is already activated
    if user_response.data.get("activated_at"):
        raise HTTPException(status_code=400, detail="Account is already activated")
    
    # Check if activation key exists and matches this user's record
    # Admin manually adds activation_key to the user's record after payment
    # So we check if this user's record has the matching key that hasn't been activated
    key_check = supabase.table("users").select("id, activation_key, activated_at").eq("id", user_id).eq("activation_key", activation_data.activation_key.upper()).is_("activated_at", "null").execute()
    
    if not key_check.data or len(key_check.data) == 0:
        # Key doesn't match this user's record or is already used
        raise HTTPException(status_code=404, detail="Invalid or already used activation key")
    
    # Key is valid and matches this user - activate the account
    update_response = supabase.table("users").update({
        "activated_at": "now()",
    }).eq("id", user_id).execute()
    
    if not update_response.data:
        raise HTTPException(status_code=500, detail="Failed to activate account")
    
    return {"message": "Account activated successfully", "activated": True}


@router.get("/status")
async def get_activation_status(user_id: str = Depends(get_user_id)):
    """
    Get activation status for the current user.
    """
    user_response = supabase.table("users").select("role, activation_key, activated_at").eq("id", user_id).single().execute()
    
    if not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = user_response.data
    is_activated = user.get("activated_at") is not None
    
    return {
        "is_activated": is_activated,
        "role": user.get("role"),
        "activated_at": user.get("activated_at"),
    }
