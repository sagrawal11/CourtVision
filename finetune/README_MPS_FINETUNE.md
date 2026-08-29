# WASB-SBDT fine-tuning on Apple MPS — de-risk report

**Verdict: FEASIBLE on MPS.** Upstream WASB-SBDT has a real training loop (it was just
commented-out + had a broken import). With ~6 small patches it trains on the Mac's MPS GPU:
our checkpoint loads as init (perfect match, 0 mismatches), forward+backward+optimizer all run
natively on MPS with **no CPU fallback**, loss is finite and trends down, and checkpoints
save + reload cleanly. Measured **~0.8 it/s** at batch 8 (512x288, 3-frame HRNet).

Everything here lives under `/tmp/wasb_finetune_work/` — the repo's `cv/` was NOT touched.

--------------------------------------------------------------------------------
## (a) Training entrypoint, how to load our init, config to fine-tune
--------------------------------------------------------------------------------
- Entrypoint: `repo/src/main.py` (Hydra). It calls `runners.select_runner(cfg).run()`.
- Trainer: `repo/src/runners/train_and_test.py::Trainer(BaseRunner)` — the actual train loop.
  Loop body (`runners/runner_utils.py::train_epoch`): `preds=model(imgs); loss=crit(preds,hms); loss.backward(); opt.step()`.
- **Upstream state**: `Trainer` was commented out in `runners/__init__.py`, and it imported a
  non-existent `runners.inference_videos` (the class actually lives in `runners/eval.py`). So
  out-of-the-box the training loop does not run even on CUDA. Both fixed in the patches.
- Model build: `models/build_model` -> `models/hrnet.py::HRNet` from `configs/model/wasb.yaml`
  (frames_in=3, frames_out=3, 512x288 in/out, out_scales=[0], ~1.49M params).
- Init loading: added `runner.resume` (path). Trainer loads it `strict=False`, stripping a
  `module.` prefix if present. Our `wasb_tennis_best.pth.tar` -> **missing=0, unexpected=0**
  (428 tensors, exact HRNet match; no `module.` prefix). No key surgery needed.
- Loss: `configs/loss/hm_wbce.yaml` (weighted BCE heatmap, scales=[0]) — correct for WASB's
  single-scale head. (Do NOT use `hm_wbce_s3` — that's for 3-scale-output models.)
- Optimizer: `configs/optimizer/adam_multistep.yaml` (Adam) — override LR to 1e-4 for fine-tune.
  (Upstream default for WASB is adadelta lr=1.0; for a warm-start you want a small Adam LR.)
- New configs added: `configs/train.yaml` (top-level) and `configs/runner/train.yaml`.

--------------------------------------------------------------------------------
## (b) On-disk data format + conversion recipe
--------------------------------------------------------------------------------
The Tennis loader (`datasets/tennis.py` + `utils/file.py::load_csv_tennis`) is TrackNet-style and
is STRICT about indexing. Per clip it needs:

```
<root_dir>/<match>/<clip>/000000.jpg 000001.jpg ...   # EVERY frame, contiguous ids 0..N-1
<root_dir>/<match>/<clip>/Label.csv                    # ONE ROW PER FRAME (dense, contiguous)
        columns:  file name,visibility,x-coordinate,y-coordinate
        - file name : "000000.jpg" (stem must parse to the frame's int id)
        - visibility: 0/1  (values in dataset.visible_flags, default [1,2], count as "ball present")
        - x,y       : ORIGINAL pixel coords; loader affine-warps to the 512x288 net input
```
Why dense/contiguous: `tennis.py` does `ball_xyvs[j] for j in range(len(csv)-frames_in+1)`,
indexing the CSV by contiguous integers starting at 0 and pairing 1:1 with sorted frame files.
A sparse or non-contiguous CSV raises KeyError / misaligns labels.

Config knobs for the loader:
`dataset.root_dir`, `dataset.train.matches=[<dir names>]`, `dataset.csv_filename=Label.csv`
(default), `dataset.ext=.jpg`, `dataset.visible_flags=[1,2]`.

### RacketVision -> WASB  (script: `convert_racketvision_to_wasb.py`)
RacketVision is `Frame,Visibility,X,Y` (X,Y original px, (0,0)=invisible) and SPARSE (~1 label/12
frames). The converter extracts frames from the mp4, renames to contiguous `%06d.jpg`, writes a
dense `Label.csv`, and optionally densifies sparse labels by **linear interpolation across visible
gaps <= --interp-max-gap** (approximation for warm-start/smoke; for production use ballistic-fit +
human-correct instead — see FINETUNE_WASB.md). Example used for the smoke test:
```
python3 convert_racketvision_to_wasb.py \
  --rv-root /tmp/racketvision_work/data \
  --out-root /tmp/wasb_finetune_work/datasets/tennis_rv \
  --clips match1/000 match10/000 match100/000 match101/000 match102/000 \
  --interp-max-gap 12 --frame-range 0 200 --jpg-quality 90
# -> 5 clips x 200 frames = 1000 frames, 92% visible after interpolation
```

### OUR fixed-cam footage -> WASB (same target format)
Produce, per clip: `<root>/<match>/<clip>/%06d.jpg` (extract every frame, e.g. ffmpeg
`-start_number 0 out/%06d.jpg`) + a dense `Label.csv` with `file name,visibility,x-coordinate,
y-coordinate`, one row per extracted frame, x/y in original pixels, visibility 1 where the ball is
labeled else 0. Densify our labels cheaply via the semi-auto loop (run WASB high-conf -> keep
high-precision hits -> short ballistic-window interpolation of <=3-frame gaps -> human-correct only
flagged frames), target ~3-8k corrected frames. If your labels are already a `frame,x,y,visibility`
CSV, just rename columns to `file name,visibility,x-coordinate,y-coordinate` (with `file name` =
`%06d.jpg`) and make it dense/contiguous. To prevent forgetting, drop RacketVision clips into the
SAME `root_dir` alongside your matches (~1:1 frames) and list them all in `dataset.train.matches`.

--------------------------------------------------------------------------------
## (c) MPS feasibility verdict + patches/gotchas/fallbacks
--------------------------------------------------------------------------------
**Feasible. No essential op lacks an MPS kernel; nothing fell back to CPU** during
forward/backward (HRNet = conv2d, batchnorm, ReLU, nearest Upsample, sigmoid; WBCE = log/mul/mean).
`PYTORCH_ENABLE_MPS_FALLBACK=1` was exported as a safety net but was not exercised. torch 2.7.1,
MPS available. No DataParallel, no OOM (model ~6 MB / 1.49M params; batch 8 @ 512x288 is tiny).

Patches (all in `mps_patches.diff`, applied to the scratch copy only):
1. `runners/__init__.py` — uncomment `Trainer`, register `'train'` runner.
2. `runners/train_and_test.py` — fix import (`inference_videos`->`eval`); allow device in
   {cuda,mps,cpu}; skip `nn.DataParallel` unless CUDA; guard `torch.cuda.empty_cache()`;
   `state_dict()` handles non-DataParallel; add `runner.resume` init-load (strict=False).
3. `runners/runner_utils.py` — `train_epoch` now does `imgs = imgs.to(device)` (CUDA path relied
   on DataParallel to scatter CPU inputs; MPS has no scatter, so this is REQUIRED).
4. `detectors/detector.py` — same device guards + `map_location='cpu'` load (only used if you turn
   on in-loop `inference_video`/`test`; off by default here).
5. `utils/utils.py::set_seed` — guard the `torch.cuda.manual_seed*` / cudnn calls behind
   `torch.cuda.is_available()`.

Gotchas:
- **First MPS step is slow (~8s)**: one-time kernel compilation. Warm up 1-2 steps before timing.
- **num_workers**: 4-6 works on macOS and improves throughput (data load was ~0.4s/it at nw=0).
  Use `dataloader.train_num_workers=6`.
- **Benign end-of-run hang**: after the final epoch the process can sit ~1-2 min in DataLoader
  worker / stdout teardown before exiting. Checkpoints are already written by then; not fatal.
- Keep in-loop `test`/`inference_video` OFF for pure fine-tuning (they pull the tracker/eval path
  and a `find_fp1` mode that needs the video-inference runner). Evaluate separately with
  `eval_racketvision.py` (the existing harness) after each stage.

--------------------------------------------------------------------------------
## (d) Smoke-test result + measured it/s + wall-clock estimate
--------------------------------------------------------------------------------
Standalone (`smoke_test_mps.py`, 60 steps, bs=8, nw=6, MPS, from our init):
- init: loaded `wasb_tennis_best.pth.tar` strict=False -> **missing=0 unexpected=0**
- speed: **60 steps / 73.5s -> 0.816 it/s (1.225 s/it)**  (~6.5 windows/s)
- loss: mean(first10)=0.000208 -> mean(last10)=0.000134, all finite (down ~36%)
- ckpt: saved 6.1 MB, reloaded missing=0 unexpected=0

Full pipeline (`main.py`, 2 epochs, bs=8, nw=4, MPS, with `runner.resume`):
- `=> loaded init weights (strict=False). missing=0, unexpected=0`
- Epoch 1 Loss 0.000149 (163.2s for 123 batches = 0.75 it/s); Epoch 2 Loss 0.000102
  (165.7s) — **loss decreases across epochs**; `checkpoint_ep1/ep2.pth.tar` both saved.

(The absolute loss is tiny because our init is already tennis-trained and the interpolated RV
labels closely match it; the point of the smoke test is loop correctness + trend + speed, which
all pass.)

Wall-clock estimates @ 0.8 it/s, bs=8 (1 window ~ 1 frame):
- **Stage-1** RV tennis warm-start (350 rallies): ~13 h (50k frames, labeled-region) to ~26 h
  (100k frames, dense) for **6 epochs**. Practical: extract only labeled-region frames and run
  3-4 epochs -> **~5-9 h**.
- **Stage-2** on our footage (3-8k frames, mixed ~1:1 with RV = 6-16k), 20 epochs:
  **~5 h (6k)** to **~14 h (16k)**. Fewer epochs (10-12) -> **~3-8 h**.
- Rule of thumb on this Mac: **~1000 frames/epoch ≈ 2.1 min**. Batch 16 may raise throughput
  further if memory allows (untested here).

These are overnight-scale on the Mac (vs ~1-4 h on a 3090/4090). If Stage-1 dense is too slow,
reduce epochs and/or extract only labeled-region frames — Stage-2 is the run that matters and it
is comfortably an afternoon/overnight job.

--------------------------------------------------------------------------------
## (e) Concrete launch commands
--------------------------------------------------------------------------------
Stage-1 (RacketVision warm-start): `bash launch_stage1_racketvision.sh`
Stage-2 (our footage):             `bash launch_stage2_ourfootage.sh`
(both scripts auto-list match dirs under their DATA_ROOT and set MPS + LR + aug). Raw form:

```
cd /tmp/wasb_finetune_work/repo/src
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 main.py --config-name=train \
  dataset.root_dir=<WASB_ROOT> 'dataset.train.matches=[<m1,m2,...>]' 'dataset.test.matches=[]' \
  dataloader.train=True dataloader.test=False dataloader.train_clip=False dataloader.test_clip=False \
  dataloader.train_num_workers=6 dataloader.sampler.train_batch_size=8 \
  runner.device=mps runner.max_epochs=<E> \
  runner.resume="<INIT_CKPT>" \
  optimizer=adam_multistep optimizer.learning_rate=1e-4 \
  loss=hm_wbce transform=full \
  output_dir=<OUT> hydra.run.dir=<OUT>
```
Stage-1 INIT = our `models/ball/wasb_tennis_best.pth.tar`; Stage-2 INIT = Stage-1's
`<OUT>/best_model.pth.tar` (falls back to our ckpt if Stage-1 skipped). Each stage writes
`checkpoint_ep<N>.pth.tar`; the final ckpt drops straight into the repo's `models/ball/` slot
(same schema, verified reloadable).

Evaluate after each stage with the existing harness (recall must stay >=0.80 on RV):
`python3 /tmp/racketvision_work/eval_racketvision.py --model wasb ...`

--------------------------------------------------------------------------------
## (f) Blockers
--------------------------------------------------------------------------------
None that block MPS fine-tuning. Notes:
- Only speed is a constraint (MPS ~0.8 it/s; overnight-scale, not CUDA-scale). Not a correctness blocker.
- In-loop eval/inference_video/find_fp1 paths are left OFF (they need the tracker + a video-inference
  runner); use the standalone eval harness. Enabling them on MPS would need extra (untested) patching.
- Real fine-tune LABEL QUALITY (not the loop) is the actual project risk: RV is sparse and naive
  interpolation is only an approximation — densify our footage with the ballistic-fit + human-correct
  loop before the Stage-2 run that counts.

## Files
- `convert_racketvision_to_wasb.py`  — RV -> WASB converter (also the template for our footage).
- `smoke_test_mps.py`                — reusable MPS smoke test (init load / speed / loss / ckpt).
- `launch_stage1_racketvision.sh`, `launch_stage2_ourfootage.sh` — fine-tune launchers.
- `mps_patches.diff`                 — all upstream patches (also already applied in `repo/`).
- `repo/`                            — patched WASB-SBDT (scratch copy; new `configs/train.yaml`,
                                       `configs/runner/train.yaml`).
- `datasets/tennis_rv/`              — the 5 converted smoke-test clips.
- `outputs/smoke1/`                  — 2-epoch run checkpoints + `main.log`.
```
