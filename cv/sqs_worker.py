"""
cv/sqs_worker.py — Long-running SQS consumer for CV analysis jobs.

Runs on the EC2 GPU worker. Polls the SQS queue for analysis jobs, launches
analysis_job.py as a subprocess for each one, and deletes the message on success.
Failed jobs are retried up to 3 times (controlled by the SQS queue's redrive
policy) before landing in the dead-letter queue.

Usage:
    python cv/sqs_worker.py

Environment variables (from /app/.env on the EC2 instance):
    SQS_QUEUE_URL             Required — SQS queue URL from Terraform outputs
    AWS_REGION_NAME           Defaults to us-east-1
    SUPABASE_URL              Required
    SUPABASE_SERVICE_ROLE_KEY Required
    PYTHONPATH                Should include the repo root (/app)
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_JOB = PROJECT_ROOT / "cv" / "analysis_job.py"

SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
AWS_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")
POLL_WAIT_SECONDS = 20  # SQS long-polling window


def process_message(body: dict) -> None:
    match_id = body["match_id"]
    frame_skip = int(body.get("frame_skip", 1))
    logger.info(f"Processing match={match_id} frame_skip={frame_skip}")

    result = subprocess.run(
        [sys.executable, str(ANALYSIS_JOB), "--match-id", match_id, "--frame-skip", str(frame_skip)],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    if result.returncode != 0:
        raise RuntimeError(f"analysis_job exited with code {result.returncode} for match={match_id}")

    logger.info(f"Analysis complete: match={match_id}")


def run():
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    logger.info(f"CV worker started — polling {SQS_QUEUE_URL}")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=POLL_WAIT_SECONDS,
                AttributeNames=["ApproximateReceiveCount"],
            )
        except ClientError as e:
            logger.error(f"SQS receive error: {e}")
            time.sleep(30)
            continue

        messages = response.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        receipt_handle = msg["ReceiptHandle"]
        receive_count = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", 1))

        try:
            body = json.loads(msg["Body"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Malformed message: {e} — deleting to avoid loop")
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            continue

        logger.info(f"Received message (attempt {receive_count}): {body}")

        try:
            process_message(body)
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            logger.info("Message deleted — job done")
        except Exception as e:
            logger.error(f"Job failed: {e}")
            # Don't delete — let SQS retry up to maxReceiveCount, then DLQ takes it


if __name__ == "__main__":
    # Validate required env vars before entering the loop
    missing = [v for v in ("SQS_QUEUE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)

    run()
