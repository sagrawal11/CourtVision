# Vercel Deployment Checklist

This guide will help you deploy the frontend to Vercel and ensure everything is configured correctly.

## ⚠️ Important: Deploy Backend First!

**Before deploying the frontend, you should deploy your backend first!**

- The frontend needs the backend URL for the `NEXT_PUBLIC_API_URL` environment variable
- See `RAILWAY_DEPLOYMENT.md` for backend deployment instructions
- We recommend using **Railway** for backend deployment (see guide)

## ✅ Pre-Deployment Checklist

### 1. Code is Ready
- [x] TypeScript builds successfully (`npm run build`)
- [x] No linter errors
- [x] All environment variables are properly configured
- [ ] **Backend is deployed** (Railway or other platform)
- [ ] **Backend URL is available**

### 2. Environment Variables Required

Before deploying to Vercel, you'll need to set these environment variables in the Vercel dashboard:

#### Required Variables:
- `NEXT_PUBLIC_SUPABASE_URL` - Your Supabase project URL
  - Example: `https://xxxxx.supabase.co`
  - Get this from: Supabase Dashboard > Settings > API > Project URL

- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Your Supabase anonymous/public key
  - Example: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
  - Get this from: Supabase Dashboard > Settings > API > anon/public key

- `NEXT_PUBLIC_API_URL` - Your backend API URL
  - For production: `https://your-backend-domain.com` (e.g., Railway, Render, etc.)
  - ⚠️ **Important**: This must be your production backend URL, not localhost!

## 🚀 Deployment Steps

### Step 1: Push to GitHub

1. Ensure all your code is committed:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Important**: Make sure your `.env.local` file is **NOT** committed (it should be in `.gitignore`)

### Step 2: Connect to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Add New Project"
3. Import your GitHub repository
4. Vercel will auto-detect it's a Next.js project

### Step 3: Configure Project Settings

1. **Root Directory**: If your frontend is in a subdirectory, set it to `frontend`
   - In Vercel: Settings > General > Root Directory > `frontend`

2. **Framework Preset**: Should auto-detect as Next.js

3. **Build Command**: `npm run build` (default)

4. **Output Directory**: `.next` (default)

5. **Install Command**: `npm install` (default)

### Step 4: Set Environment Variables

1. In Vercel project settings, go to **Settings > Environment Variables**

2. Add each required variable:
   - Click "Add New"
   - Enter variable name (e.g., `NEXT_PUBLIC_SUPABASE_URL`)
   - Enter variable value
   - Select environments: **Production**, **Preview**, and **Development**
   - Click "Save"

3. Repeat for all three required variables:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `NEXT_PUBLIC_API_URL`

### Step 5: Deploy

1. Click "Deploy" button
2. Wait for the build to complete
3. Once deployed, Vercel will provide you with a URL like: `https://your-app.vercel.app`

## 🔧 Post-Deployment Checklist

### 1. Test the Deployment

- [ ] Visit your Vercel URL and verify the landing page loads
- [ ] Test authentication (sign up / sign in)
- [ ] Verify API calls work (check browser console for errors)
- [ ] Test protected routes (dashboard, teams, etc.)

### 2. Update Backend CORS

Make sure your backend allows requests from your Vercel domain:

**In your backend (FastAPI)**, update `ALLOWED_ORIGINS`:
```python
ALLOWED_ORIGINS = "https://your-app.vercel.app,http://localhost:3000"
```

Or in your `.env`:
```env
ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

### 3. Update Supabase Settings (if needed)

If you have any domain restrictions in Supabase:
1. Go to Supabase Dashboard > Settings > API
2. Add your Vercel domain to allowed origins (if required)

### 4. Custom Domain (Optional)

To use a custom domain:
1. In Vercel project: Settings > Domains
2. Add your custom domain
3. Follow DNS configuration instructions
4. Update `NEXT_PUBLIC_API_URL` if needed (should still point to backend)

## 🐛 Troubleshooting

### Build Fails

**Error: "Environment variable not found"**
- Make sure all `NEXT_PUBLIC_*` variables are set in Vercel
- Variables must be set for Production, Preview, AND Development environments

**Error: "TypeScript errors"**
- Run `npm run build` locally first to catch errors
- All TypeScript errors should be fixed before deployment

### Runtime Errors

**Error: "Failed to fetch" from API**
- Check that `NEXT_PUBLIC_API_URL` is set correctly
- Verify backend is deployed and accessible
- Check backend CORS settings include your Vercel domain
- Check browser console for specific error messages

**Error: "Supabase authentication failed"**
- Verify `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` are correct
- Check Supabase project is active (not paused)
- Verify Supabase Auth settings allow your Vercel domain

### Middleware Warning

You may see a warning: `"The 'middleware' file convention is deprecated"`
- This is a Next.js warning, not an error
- The app will still work, but consider migrating to the new `proxy` convention in the future
- Current middleware.ts file is fine for deployment

## 📝 Notes

1. **Environment Variables**: Only `NEXT_PUBLIC_*` variables are exposed to the browser. Never put secrets in `NEXT_PUBLIC_*` variables.

2. **Backend URL**: The `NEXT_PUBLIC_API_URL` must be your production backend URL. Make sure your backend is deployed before deploying the frontend, or API calls will fail.

3. **Supabase**: Your Supabase project should already be set up and the database schema should be deployed.

4. **Automatic Deployments**: Vercel will automatically deploy on every push to your main branch (if configured).

## 🔗 Useful Links

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Deployment](https://nextjs.org/docs/deployment)
- [Environment Variables in Vercel](https://vercel.com/docs/concepts/projects/environment-variables)
