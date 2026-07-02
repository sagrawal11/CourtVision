from fastapi import Header, HTTPException
import os
from supabase import create_client

# Initialize Supabase for auth verification
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

if supabase_url and supabase_anon_key:
    supabase_auth = create_client(supabase_url, supabase_anon_key)
else:
    supabase_auth = None


async def get_user_id(authorization: str = Header(None)):
    """Extract user ID from authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract token (assuming "Bearer <token>" format)
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    
    if not supabase_auth:
        raise HTTPException(status_code=500, detail="Auth not configured")
    
    # Verify token and get user
    try:
        user_response = supabase_auth.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_response.user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
