#!/usr/bin/env bash
# STAGE 2 — fine-tune WASB (HRNet) on OUR footage (indoor + outdoor) + RacketVision ~1:1, MPS.
# Init from the Stage-1 ep4 checkpoint (inherits its outdoor-clutter rejection); goal =
# recover our-outdoor recall while keeping clutter dead. runner.resume loads WEIGHTS ONLY
# (train_and_test.py:91-94), so optimizer + LR schedule start fresh at epoch 0.
# Mix (mix_stage2): indoor1 700 + outdoor1_seg10845 1351 + 8 RV matches 1985 = our:RV ~1:0.97.
set -euo pipefail

REPO=/tmp/wasb_finetune_work/repo/src
DATA_ROOT=/tmp/wasb_finetune_work/datasets/mix_stage2
INIT="/Users/sarthak/Desktop/App Projects/tennis_analytics/models/ball/wasb_stage1_ep4.pth.tar"
OUT=/tmp/wasb_finetune_work/outputs/stage2_ours
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
  runner.max_epochs=12 \
  runner.resume="$INIT" \
  runner.test.run=False runner.inference_video.run=False \
  optimizer=adam_multistep optimizer.learning_rate=1e-4 \
  "optimizer.scheduler.stepsize=[6,9]" optimizer.scheduler.gamma=0.1 \
  loss=hm_wbce \
  transform=full \
  output_dir="$OUT" hydra.run.dir="$OUT"
