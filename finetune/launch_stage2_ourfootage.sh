#!/usr/bin/env bash
# STAGE 2 — fine-tune WASB (HRNet) on OUR fixed-cam footage on Apple MPS.
# Init from the stage-1 checkpoint (or directly from our tennis checkpoint).
# Mix RacketVision frames in ~1:1 to prevent forgetting broadcast generalization
# (put both RV match dirs and our-footage match dirs under the same DATA_ROOT and
#  list them all in matches).
set -euo pipefail

REPO=/tmp/wasb_finetune_work/repo/src
DATA_ROOT=/tmp/wasb_finetune_work/datasets/mix_stage2      # our clips (+ RV clips for replay) in WASB layout
# Stage-1 wrote per-epoch checkpoints only (NO best_model.pth.tar — in-loop eval was off).
# ep4 is persisted in the repo; use it directly. (See finetune/launch_stage2_run.sh for the
# actual tested Stage-2 config: 12 epochs, LR drop [6,9], test/infer off.)
STAGE1_CKPT="/Users/sarthak/Desktop/App Projects/tennis_analytics/models/ball/wasb_stage1_ep4.pth.tar"
OUT=/tmp/wasb_finetune_work/outputs/stage2_ours
MATCHES=$(python3 -c "import os;print(','.join(sorted(d for d in os.listdir('$DATA_ROOT') if os.path.isdir(os.path.join('$DATA_ROOT',d)))))")

cd "$REPO"
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 main.py --config-name=train \
  dataset.root_dir="$DATA_ROOT" \
  "dataset.train.matches=[$MATCHES]" \
  "dataset.test.matches=[]" \
  dataloader.train=True dataloader.test=False \
  dataloader.train_clip=False dataloader.test_clip=False \
  dataloader.train_num_workers=6 dataloader.test_num_workers=0 \
  dataloader.sampler.train_batch_size=8 \
  runner.device=mps \
  runner.max_epochs=20 \
  runner.resume="$STAGE1_CKPT" \
  optimizer=adam_multistep optimizer.learning_rate=1e-4 \
  "optimizer.scheduler.stepsize=[12,17]" optimizer.scheduler.gamma=0.1 \
  loss=hm_wbce \
  transform=full \
  output_dir="$OUT" hydra.run.dir="$OUT"
echo "stage-2 done. deploy $OUT/best_model.pth.tar into the repo's models/ball/ slot."
