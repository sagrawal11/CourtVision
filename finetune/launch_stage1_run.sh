#!/usr/bin/env bash
# STAGE 1 (actual overnight run) — warm-start WASB (HRNet) on the RacketVision tennis
# TRAIN subset (120 clips) on Apple MPS. Init from our tennis checkpoint; LR 1e-4;
# single-scale heatmap WBCE loss; full aug; 4 epochs. All in-loop eval/inference OFF.
#
# Based on launch_stage1_racketvision.sh (prior agent), with: max_epochs 6->4,
# scheduler stepsize [4,5]->[3] to suit a 4-epoch schedule, and this run's own OUT dir.
set -euo pipefail

REPO=/tmp/wasb_finetune_work/repo/src
DATA_ROOT=/tmp/wasb_finetune_work/datasets/tennis_rv_train
INIT="/Users/sarthak/Desktop/App Projects/tennis_analytics/models/ball/wasb_tennis_best.pth.tar"
OUT=/tmp/wasb_finetune_work/outputs/stage1_rv
# comma-separated list of match dirs under DATA_ROOT:
MATCHES=$(python3 -c "import os;print(','.join(sorted(d for d in os.listdir('$DATA_ROOT') if os.path.isdir(os.path.join('$DATA_ROOT',d)))))")

cd "$REPO"
PYTORCH_ENABLE_MPS_FALLBACK=1 exec python3 main.py --config-name=train \
  dataset.root_dir="$DATA_ROOT" \
  "dataset.train.matches=[$MATCHES]" \
  "dataset.test.matches=[]" \
  dataloader.train=True dataloader.test=False \
  dataloader.train_clip=False dataloader.test_clip=False \
  dataloader.train_num_workers=6 dataloader.test_num_workers=0 \
  dataloader.sampler.train_batch_size=8 \
  runner.device=mps \
  runner.max_epochs=4 \
  runner.resume="$INIT" \
  runner.test.run=False runner.inference_video.run=False \
  optimizer=adam_multistep optimizer.learning_rate=1e-4 \
  "optimizer.scheduler.stepsize=[3]" optimizer.scheduler.gamma=0.1 \
  loss=hm_wbce \
  transform=full \
  output_dir="$OUT" hydra.run.dir="$OUT"
