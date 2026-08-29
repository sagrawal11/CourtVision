# Tennis Ball Tracking — SOTA Survey & Recommendation (2022–2026)

_Produced 2026-08-25 by a deep-research pass (fan-out web search + adversarial cross-verification). Feeds the decision on how to make ball tracking "usable anywhere" (fixed-camera venues, indoor + outdoor)._

**Bottom line:** WASB (our current detector) is still among the strongest _published_ per-frame tennis ball detectors. Its outdoor failure is a **domain-shift + clutter problem, not an architecture ceiling** — the WASB paper never evaluated outdoors, and its false positives are explicitly "noisy background" clutter, exactly what we see. **No off-the-shelf model solves our outdoor problem without our own labeled fixed-camera data.** Highest-ROI move: **fine-tune WASB on our own footage + add a trajectory-gating layer** — not switching architectures. The 2024–2026 "SOTA" (TrackNetV4/V5) is largely over-claimed and badminton-benchmarked; the genuinely useful new work is **RacketVision** (a real, downloadable tennis benchmark) and **TOTNet** (occlusion-aware temporal).

Independent triangulation of our production experience: RacketVision (AAAI 2026) benchmarks WASB tennis at **recall 0.803 / precision 0.937** — the 0.80 recall reproduces our observed ~81% on indoor ball-present frames.

## Comparison (key rows)

| Model | Arch | Input / frames | Best reported (dataset) | Speed | Tennis weights / license | Notes |
|---|---|---|---|---|---|---|
| **WASB** (BMVC'23, ours) | HRNet heatmap + online quadratic-motion tracker | 288×512, 3 frames | F1 95.6 (own set); **recall 0.803 / prec 0.937** (RacketVision) | 35 fps V100; ~40ms MPS | Yes, MIT | FPs on noisy background; **no outdoor eval** |
| TrackNetV2 | U-Net MIMO (3→3) | 512×288, 3 | tennis F1 89.4 (WASB's table) | 17–85 fps | in WASB zoo | occlusion acc 0.84→0.49→0.00 (vis/part/full) |
| TrackNetV3 | V2 + trajectory-rectify (InpaintNet) | seq 8 | 97.51% **badminton, not tennis**; repro F1 0.809 | ~25 fps | MIT, **shuttlecock only** | rectifier helps trajectories not raw detection |
| TrackNetV4 (ICASSP'25) | V2/V3 + motion-attention | 512×288, 3 | **no absolute tennis metrics, no speed** | — | MIT | unverifiable on our axes |
| TrackNetV5 (preprint) | motion-decouple + Transformer refine | 512×288, 3 | paper F1 0.986; **repro F1 0.773** | 114 fps T4 (claimed) | no official weights | large paper-vs-repro gap |
| **TOTNet** (Aug'25) | 3D U-Net (+optional RAFT flow), occlusion-aware | 288×512, **5 frames** | tennis occ **0.95/0.61/0.67** vs WASB 0.92/0.52/**0.17**; RMSE 6.07 vs 16.58 | 12–28 fps (offline) | CC-BY-4.0 | **best evidence vs our occlusion failure**; single-source |
| **MS-TrackNetV3** (RacketVision, AAAI'26) | TrackNetV3 trained **multi-sport** | 512×288 | tennis **P 0.945 / R 0.880 / mAP 81.9** (+19 mAP over single-sport) | — | code+data public | multi-sport training > single-sport (independently benchmarked); **beats WASB on recall** |
| YOLO11 / RTMDet | single-frame detectors | 640 | tennis recall **0.42–0.48** | fast | Ultralytics | generic detectors have poor ball recall; use only as candidate generator |

## Ranked recommendation

**Path 1 (RECOMMENDED) — Fine-tune WASB on our own fixed-camera footage + trajectory-gating layer.** WASB is already top-tier; outdoor collapse is domain shift. Fine-tune on a few thousand of our own outdoor/indoor frames (highest ROI, we already run the model). Add a ballistic/trajectory-consistency gate (reject detections that don't fit a short parabola; interpolate 1–3 frame gaps) to kill clutter FPs + smooth trajectories. Won't fix full-occlusion frames; keeps ~40ms budget.

**Path 2 — RacketVision recipe: multi-sport TrackNet-family (MS-TrackNetV3).** Most credible new result (AAAI'26 Oral, public data+code): multi-sport joint training lifts tennis ball mAP ~19pts and recall to 0.88 (beats WASB on our weak axis). Pre-train on RacketVision, fine-tune on our footage; get a free eval harness. Best bet if fine-tuned WASB doesn't clear the recall bar.

**Path 3 — TOTNet-style occlusion-aware temporal for hard frames (targeted).** Only paper with direct evidence vs our failure mode (full-occlusion 0.17→0.67). Offline/slower (5-frame 3D conv), single-source — use as an offline enhancer or architectural inspiration, not a real-time drop-in.

**Do NOT:** switch to single-frame YOLO as primary (recall 0.42–0.48); chase TrackNetV4/V5 headline numbers (over-claimed, badminton, big paper-vs-repro gaps).

## Data strategy (cheapest path to ball labels)

1. **Bootstrap free:** RacketVision — 431 tennis clips / 150,399 frames / **21,544 ball annotations**, MIT/CC, on Hugging Face — pre-train + eval. Plus WASB's own tennis set (14,160 train). All broadcast-style → transfer partially, **cannot substitute for our own fixed-cam outdoor frames**.
2. **Semi-automated labeling on OUR footage (the actual fix):** run WASB at high confidence → keep high-precision detections → fit short-window ballistic trajectories + interpolate gaps (auto-labels easy + interpolatable frames) → **human correction only on flagged frames** (trajectory outliers, low-conf, near-net/bounce). ~5–10× cheaper than frame-by-frame. Target ~3,000–8,000 corrected fixed-cam frames spanning sun/shadow/low-contrast, prioritizing outdoor+cluttered. Doubles as our eval GT.
3. **Synthetic:** proven for 3D trajectory lifting, not 2D appearance under sun/shadow — low priority.

Estimate: RacketVision pre-train + ~5k semi-auto-labeled fixed-cam frames → fine-tuned WASB should recover ~90% F1 / 80%+ recall outdoors, trajectory gate cutting clutter FPs.

## Verified over-claims / caveats
- TrackNetV3 97.51% is **badminton**, not tennis (repro 0.707 acc). TrackNetV5 tennis F1 0.986 → repro 0.773; no official weights. TrackNetV4 reports no absolute tennis metrics or speed. TOTNet's WASB-beating numbers are single-source/self-reported and framed offline. Every tennis-ball paper evaluates on its own small in-domain set; almost none test outdoor/cross-venue; independent repros land 10–27 pts below paper claims.

## Sources
- WASB: https://arxiv.org/abs/2311.05237 · https://github.com/nttcom/WASB-SBDT
- RacketVision (AAAI'26; public data+code): https://arxiv.org/abs/2511.17045 · https://github.com/OrcustD/RacketVision · https://huggingface.co/datasets/linfeng302/RacketVision
- TOTNet: https://arxiv.org/abs/2508.09650 · https://github.com/AugustRushG/TOTNet
- TrackNetV3: https://github.com/qaz812345/TrackNetV3 · V4: https://arxiv.org/abs/2409.14543 · V5: https://arxiv.org/html/2512.02789
- Independent TrackNet V1–V5 reproduction: https://github.com/AnInsomniacy/tracknet-series-pytorch
