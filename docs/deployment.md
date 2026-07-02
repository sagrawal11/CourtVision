# Deployment

Courtvision runs four services in production:

| Layer | Host | Notes |
|---|---|---|
| Frontend | Vercel | Next.js, auto-deploys on push |
| Backend API | AWS ECS Fargate | FastAPI, CPU-only, behind ALB |
| CV Worker | AWS EC2 g4dn.xlarge | GPU, polls SQS queue |
| Database / Auth / Storage | Supabase Pro | Postgres + RLS + Storage bucket |

---

## Local development

```bash
# Backend (no GPU, no SQS — runs CV jobs inline via subprocess)
source tennis_env/bin/activate
pip install -r backend/requirements.txt
./start_backend.sh        # uvicorn on :8000

# Frontend
./start_frontend.sh       # next dev on :3000
```

Local dev deliberately skips SQS: if `SQS_QUEUE_URL` is not set, `_dispatch_analysis`
falls back to `subprocess.Popen` on the same machine.

---

## Prerequisites

- AWS CLI configured (`aws configure`) with an IAM user that has `AdministratorAccess`
  (or scoped permissions for ECS, ECR, SQS, EC2, IAM, S3)
- Terraform ≥ 1.5 (`brew install terraform`)
- Docker
- An EC2 key pair created in `us-east-1` (optional, for SSH access to the CV worker)

---

## First-time AWS setup

### 1. Bootstrap Terraform state (optional but recommended)

Before running Terraform, create an S3 bucket + DynamoDB table for remote state:

```bash
aws s3api create-bucket --bucket courtvision-terraform-state --region us-east-1
aws s3api put-bucket-versioning --bucket courtvision-terraform-state \
    --versioning-configuration Status=Enabled
aws dynamodb create-table \
    --table-name courtvision-terraform-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

Then uncomment the `backend "s3"` block in `infra/main.tf` and run `terraform init`.

### 2. Provision infrastructure

```bash
cd infra

# Set sensitive vars (never put these in .tfvars committed to git)
export TF_VAR_supabase_url="https://xxx.supabase.co"
export TF_VAR_supabase_service_role_key="eyJ..."
export TF_VAR_supabase_anon_key="eyJ..."
export TF_VAR_frontend_url="https://your-app.vercel.app"
export TF_VAR_key_pair_name="your-key-pair"   # optional

terraform init
terraform plan
terraform apply
```

Note the outputs — you'll need them in the next steps:
- `backend_url` → set as `NEXT_PUBLIC_API_URL` in Vercel
- `ecr_backend_repo` → Docker push target
- `sqs_queue_url` → set on EC2 worker
- `models_s3_bucket` → upload model weights here
- `cv_worker_instance_id` → for SSM/SSH access

### 3. Upload model weights to S3

```bash
# From repo root — models/ is git-ignored, assumed present locally
aws s3 sync models/ s3://$(terraform -chdir=infra output -raw models_s3_bucket)/models/
```

### 4. Build and push the backend Docker image

```bash
REPO=$(terraform -chdir=infra output -raw ecr_backend_repo)
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker build -f Dockerfile.backend -t $REPO:latest .
docker push $REPO:latest

# Force ECS to pick up the new image
aws ecs update-service \
    --cluster courtvision \
    --service courtvision-backend \
    --force-new-deployment \
    --region $REGION
```

### 5. Set up the CV worker on EC2

Connect via SSM (no inbound SSH port needed):
```bash
INSTANCE_ID=$(terraform -chdir=infra output -raw cv_worker_instance_id)
aws ssm start-session --target $INSTANCE_ID --region us-east-1
```

On the instance:
```bash
# Clone the repo (or copy the code)
sudo git clone https://github.com/sagrawal11/courtvision /app
cd /app

# Create .env with production values
sudo tee /app/.env > /dev/null <<EOF
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_ANON_KEY=eyJ...
SQS_QUEUE_URL=<sqs_queue_url from terraform output>
AWS_REGION_NAME=us-east-1
PYTHONPATH=/app
EOF

# Sync model weights from S3 (user data already ran this at boot)
aws s3 sync s3://<models_s3_bucket>/models/ /app/models/

# Start the worker
sudo systemctl start cv-worker
sudo systemctl status cv-worker

# Tail logs
sudo journalctl -fu cv-worker
```

### 6. Deploy the frontend to Vercel

```bash
cd frontend
npx vercel --prod
```

Set these environment variables in the Vercel dashboard:
```
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_PUBLIC_API_URL=http://<backend_url from terraform output>
```

After the Vercel URL is known, update Terraform:
```bash
export TF_VAR_frontend_url="https://your-app.vercel.app"
terraform apply   # updates ALLOWED_ORIGINS in ECS task env
```

---

## Supabase setup

1. Create a project on [supabase.com](https://supabase.com), then create the
   `match-videos` Storage bucket manually in the Storage section.
2. In the SQL editor, run `supabase/schema.sql` (the complete consolidated
   schema), then `supabase/rls_policies.sql` to apply Row-Level Security.
3. Copy the project URL, anon key, and service-role key into the env vars above.

---

## Ongoing deployments

### Deploy new backend version

```bash
REPO=$(terraform -chdir=infra output -raw ecr_backend_repo)
docker build -f Dockerfile.backend -t $REPO:latest .
docker push $REPO:latest
aws ecs update-service --cluster courtvision --service courtvision-backend \
    --force-new-deployment --region us-east-1
```

### Update CV worker code

```bash
INSTANCE_ID=$(terraform -chdir=infra output -raw cv_worker_instance_id)
aws ssm start-session --target $INSTANCE_ID --region us-east-1
# On the instance:
cd /app && sudo git pull && sudo systemctl restart cv-worker
```

---

## Cost management

| Resource | Cost | Notes |
|---|---|---|
| ECS Fargate (0.5 vCPU / 1 GB) | ~$14/month | Always on |
| EC2 g4dn.xlarge | ~$0.53/hr on-demand | **Stop when not processing** |
| SQS | ~$0 | < 1M req/month free tier |
| ALB | ~$16/month | Always on |
| Supabase Pro | $25/month | — |

**The CV worker is the biggest cost lever.** Stop it when idle:
```bash
aws ec2 stop-instances --instance-ids <cv_worker_instance_id> --region us-east-1
aws ec2 start-instances --instance-ids <cv_worker_instance_id> --region us-east-1
```

Or enable Spot pricing by uncommenting `instance_market_options` in
`infra/modules/ec2-worker/main.tf` — saves up to 70% but the instance can be
interrupted (the SQS message will become visible again after the visibility
timeout and be retried).

---

## Monitoring

- **Backend logs**: CloudWatch → `/ecs/courtvision/backend`
- **CV worker logs**: SSM into the instance → `sudo journalctl -fu cv-worker`
- **Failed jobs**: SQS → Dead-letter queue `courtvision-analysis-jobs-dlq`
- **ECS service health**: AWS Console → ECS → courtvision cluster
