# Railway Backend Deployment Guide

This guide will help you deploy your FastAPI backend to Railway.

## 🚂 Why Railway?

- **Easy Setup**: Connects directly to GitHub
- **Free Tier**: $5/month credit (usually enough for small apps)
- **Automatic Deployments**: Deploys on every push
- **Simple Configuration**: Easy environment variable management
- **HTTPS Included**: Automatic SSL certificates
- **Great for FastAPI**: Built-in Python support

## ✅ Pre-Deployment Checklist

### 1. Code is Ready
- [x] Backend code is committed to GitHub
- [x] `requirements.txt` is up to date
- [x] No hardcoded credentials
- [x] Environment variables are properly used

### 2. Environment Variables Needed

Before deploying, prepare these environment variables:

#### Required Variables:
- `SUPABASE_URL` - Your Supabase project URL
  - Example: `https://xxxxx.supabase.co`
  - Get from: Supabase Dashboard > Settings > API > Project URL

- `SUPABASE_SERVICE_ROLE_KEY` - Your Supabase service role key (SECRET!)
  - Example: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
  - ⚠️ **Keep this secret!** Never commit to git
  - Get from: Supabase Dashboard > Settings > API > service_role key

- `SUPABASE_ANON_KEY` - Your Supabase anonymous/public key
  - Example: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
  - Get from: Supabase Dashboard > Settings > API > anon/public key

- `ALLOWED_ORIGINS` - Comma-separated list of allowed origins for CORS
  - For production: `https://your-frontend.vercel.app,http://localhost:3000`
  - You can add your Vercel URL after frontend deployment
  - Format: `origin1,origin2,origin3` (no spaces)

#### Optional Variables:
- `API_PORT` - Port for the server (Railway sets this automatically via `PORT` env var)
- `ENVIRONMENT` - Set to `production` (optional, for environment-specific logic)

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure backend is in the root or a subdirectory**
   - Your backend code is in the `/backend` directory
   - Railway can deploy from a subdirectory

2. **Verify `requirements.txt` exists**
   - Should be in `/backend/requirements.txt`
   - Should include all dependencies

3. **Commit and push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

### Step 2: Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (recommended for easy repo connection)
3. Complete the onboarding process

### Step 3: Create a New Project

1. Click **"New Project"** in Railway dashboard
2. Select **"Deploy from GitHub repo"**
3. Choose your repository
4. Railway will auto-detect it's a Python project

### Step 4: Configure the Service

1. **Set Root Directory**:
   - Click on your service
   - Go to **Settings > Service**
   - Set **Root Directory** to `backend`
   - This tells Railway where your Python code is

2. **Configure Start Command**:
   - Go to **Settings > Service**
   - Set **Start Command** to:
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - Railway provides `$PORT` environment variable automatically
   - `--host 0.0.0.0` allows external connections

3. **Set Python Version** (Optional but recommended):
   - Go to **Settings > Variables**
   - Add variable: `PYTHON_VERSION` = `3.11` (or your preferred version)
   - Railway will use this Python version

### Step 5: Set Environment Variables

1. Go to **Variables** tab in your Railway service

2. Click **"New Variable"** and add each variable:

   **SUPABASE_URL**
   - Name: `SUPABASE_URL`
   - Value: Your Supabase project URL
   - Example: `https://xxxxx.supabase.co`

   **SUPABASE_SERVICE_ROLE_KEY**
   - Name: `SUPABASE_SERVICE_ROLE_KEY`
   - Value: Your Supabase service role key (keep secret!)
   - ⚠️ Make sure this is the **service_role** key, not the anon key

   **SUPABASE_ANON_KEY**
   - Name: `SUPABASE_ANON_KEY`
   - Value: Your Supabase anonymous key

   **ALLOWED_ORIGINS**
   - Name: `ALLOWED_ORIGINS`
   - Value: `http://localhost:3000` (you'll update this after frontend deployment)
   - Later, update to: `https://your-frontend.vercel.app,http://localhost:3000`

3. **Important**: Railway automatically sets `PORT` variable - don't override it

### Step 6: Deploy

1. Railway will automatically start deploying
2. Watch the build logs in the **Deployments** tab
3. Wait for deployment to complete (usually 2-5 minutes)

### Step 7: Get Your Backend URL

1. Once deployed, Railway will provide a public URL
2. It will look like: `https://your-service-name.up.railway.app`
3. Copy this URL - you'll need it for:
   - Frontend `NEXT_PUBLIC_API_URL` environment variable
   - Testing your API

4. **Test the deployment**:
   - Visit: `https://your-service-name.up.railway.app/health`
   - Should return: `{"status": "healthy"}`
   - Visit: `https://your-service-name.up.railway.app/docs`
   - Should show FastAPI Swagger documentation

## 🔧 Post-Deployment Checklist

### 1. Update CORS Settings

After you deploy your frontend to Vercel:

1. Go to Railway > Your Service > Variables
2. Find `ALLOWED_ORIGINS`
3. Update the value to include your Vercel URL:
   ```
   https://your-frontend.vercel.app,http://localhost:3000
   ```
4. Railway will automatically redeploy with the new environment variable

### 2. Update Frontend Environment Variable

1. Go to Vercel > Your Project > Settings > Environment Variables
2. Find or add `NEXT_PUBLIC_API_URL`
3. Set value to your Railway URL: `https://your-service-name.up.railway.app`
4. Redeploy your frontend (Vercel will do this automatically)

### 3. Test the Integration

1. Visit your Vercel frontend URL
2. Try to sign up/sign in
3. Check browser console for any API errors
4. Verify API calls are working

## 🐛 Troubleshooting

### Build Fails

**Error: "No module named 'xxx'"**
- Check `requirements.txt` includes all dependencies
- Verify all packages are listed with correct versions

**Error: "Could not find a version that satisfies the requirement"**
- Some packages might not be compatible with Railway's Python version
- Try specifying a Python version: Add `PYTHON_VERSION=3.11` variable

**Error: "Module not found"**
- Make sure Root Directory is set to `backend`
- Verify file structure is correct

### Runtime Errors

**Error: "Supabase credentials not configured"**
- Check all Supabase environment variables are set in Railway
- Verify variable names match exactly (case-sensitive)
- Make sure you're using `SUPABASE_SERVICE_ROLE_KEY` (not `SUPABASE_ANON_KEY`)

**Error: "CORS error" in browser**
- Check `ALLOWED_ORIGINS` includes your Vercel URL
- Format should be: `https://your-app.vercel.app,http://localhost:3000`
- No spaces between origins (just commas)

**Error: "502 Bad Gateway" or service won't start**
- Check deployment logs in Railway
- Verify start command is correct: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Check logs for Python errors

**Service keeps restarting**
- Check logs for errors
- Verify all required environment variables are set
- Check if there are any Python syntax errors

### Testing API

**Health check fails:**
```bash
curl https://your-service-name.up.railway.app/health
```

**Check API docs:**
- Visit: `https://your-service-name.up.railway.app/docs`
- Should show Swagger UI with all your endpoints

**Test authentication:**
- Try calling an authenticated endpoint with a valid token
- Check logs if it fails

## 📝 Important Notes

1. **Service Role Key**: The `SUPABASE_SERVICE_ROLE_KEY` is very sensitive. Never commit it to git. Only set it in Railway's environment variables.

2. **Port**: Railway automatically sets the `PORT` environment variable. Don't set it manually. Use `$PORT` in your start command.

3. **Auto-Deployments**: Railway automatically deploys when you push to your connected branch (usually `main`).

4. **Free Tier Limits**: 
   - $5/month credit
   - Service sleeps after inactivity (wakes up on first request)
   - First request after sleep may be slow (cold start)

5. **Custom Domain**: Railway allows custom domains (paid feature). The default `.up.railway.app` domain works fine for most use cases.

6. **Logs**: Always check Railway logs when debugging. They show Python errors, environment variable issues, and deployment problems.

## 🔄 Updating the Backend

After making changes:

1. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update backend"
   git push origin main
   ```

2. Railway will automatically detect the push and redeploy

3. Check deployment status in Railway dashboard

4. Test your changes on the deployed URL

## 🔗 Useful Links

- [Railway Documentation](https://docs.railway.app/)
- [Railway Python Guide](https://docs.railway.app/guides/python)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

## 💡 Alternative Platforms

If Railway doesn't work for you, here are alternatives:

### Render
- Similar to Railway
- Free tier with some limitations
- [render.com](https://render.com)

### Fly.io
- Docker-based deployment
- Good for more control
- [fly.io](https://fly.io)

### PythonAnywhere
- Python-focused hosting
- Free tier available
- [pythonanywhere.com](https://www.pythonanywhere.com)
