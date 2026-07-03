"""cv/tools/court_keypoint_editor.py — generate a standalone court-keypoint editor.

The camera is static, so the 14 court keypoints are placed ONCE per video and
reused for the whole clip. This grabs a frame from a video and writes a
self-contained HTML editor (no server, no dependencies) where you click the 14
points in the exact order the pipeline expects, then download keypoints.json.

    python cv/tools/court_keypoint_editor.py --video "tests/Indoor Match 1 15.53.25.mp4"
    # open the printed .html, place the 14 points, download keypoints.json, then:
    python cv/tools/evaluate_pipeline.py \
        --annotations "cv/training_data/Indoor Match 1 15.53.25_annotations.csv" \
        --video "tests/Indoor Match 1 15.53.25.mp4" --keypoints keypoints.json

The exported JSON is a list of 14 [x, y] pairs in NATIVE frame pixels, ordered
to match cv.detection.court_detector.CourtReference.key_points (which
pipeline._build_homography maps to by index). DO NOT reorder them.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

# Labels for the 14 keypoints, in CourtReference.key_points order. "Far" = top of
# frame (away from camera), "Near" = bottom (closest to camera).
KEYPOINT_LABELS = [
    "Far baseline — LEFT doubles corner",     # 0
    "Far baseline — RIGHT doubles corner",    # 1
    "Near baseline — LEFT doubles corner",    # 2
    "Near baseline — RIGHT doubles corner",   # 3
    "Far baseline — LEFT singles corner",     # 4
    "Near baseline — LEFT singles corner",    # 5
    "Far baseline — RIGHT singles corner",    # 6
    "Near baseline — RIGHT singles corner",   # 7
    "Far service line — LEFT (singles)",      # 8
    "Far service line — RIGHT (singles)",     # 9
    "Near service line — LEFT (singles)",     # 10
    "Near service line — RIGHT (singles)",    # 11
    "Far service line — CENTER",              # 12
    "Near service line — CENTER",             # 13
]

# Court lines between keypoints (indices in the order above), for visual validation.
EDGES = [
    [0, 4], [4, 6], [6, 1],          # far baseline
    [2, 5], [5, 7], [7, 3],          # near baseline
    [0, 2], [1, 3],                  # doubles sidelines
    [4, 8], [8, 10], [10, 5],        # left singles sideline
    [6, 9], [9, 11], [11, 7],        # right singles sideline
    [8, 12], [12, 9],                # far service line
    [10, 13], [13, 11],              # near service line
    [12, 13],                        # center service line
]

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Court Keypoint Editor — __NAME__</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:system-ui,sans-serif; display:flex; }
  #stage { position:relative; flex:1; overflow:auto; }
  #wrap { position:relative; display:inline-block; }
  #frame { display:block; max-width:none; }
  #ov { position:absolute; left:0; top:0; cursor:crosshair; }
  #side { width:320px; padding:16px; background:#1a1a1a; border-left:1px solid #333; height:100vh; overflow:auto; box-sizing:border-box; }
  h1 { font-size:15px; margin:0 0 8px; }
  p.hint { font-size:12px; color:#999; line-height:1.4; }
  ol { padding-left:0; list-style:none; margin:12px 0; font-size:12px; }
  li { padding:6px 8px; border-radius:6px; margin-bottom:3px; cursor:pointer; border:1px solid transparent; }
  li .n { display:inline-block; width:20px; color:#50C878; font-weight:700; }
  li.active { background:#50C878; color:#000; } li.active .n { color:#000; }
  li.done { color:#888; } li.done .n { color:#3da860; }
  button { width:100%; padding:10px; margin-top:8px; border:none; border-radius:6px; font-weight:600; cursor:pointer; }
  #export { background:#50C878; color:#000; } #export:disabled { background:#333; color:#777; cursor:not-allowed; }
  #reset { background:#333; color:#eee; }
  #zoom { display:flex; gap:6px; } #zoom button { background:#333; color:#eee; }
</style></head>
<body>
<div id="stage"><div id="wrap"><img id="frame" src="data:image/jpeg;base64,__IMG__"><canvas id="ov"></canvas></div></div>
<div id="side">
  <h1>Court Keypoint Editor</h1>
  <p class="hint">Click the image to place the highlighted point, then it advances.
  Click a point below to re-select it; drag a placed dot to nudge. "Far" = top of
  frame, "Near" = bottom (closest to camera). Place all 14, then Export.</p>
  <div id="zoom"><button onclick="setZoom(z/1.25)">−</button><button onclick="setZoom(z*1.25)">+</button><button onclick="setZoom(fit)">Fit</button></div>
  <ol id="list"></ol>
  <button id="export" disabled onclick="doExport()">Download keypoints.json</button>
  <button id="reset" onclick="if(confirm('Clear all points?')){pts=Array(14).fill(null);active=0;render();}">Reset</button>
</div>
<script>
const LABELS = __LABELS__, EDGES = __EDGES__, W = __WIDTH__, H = __HEIGHT__;
const img = document.getElementById('frame'), ov = document.getElementById('ov'), wrap = document.getElementById('wrap');
const ctx = ov.getContext('2d');
let pts = Array(14).fill(null), active = 0, z = 1, fit = 1, drag = -1;

function setZoom(nz){ z = Math.max(0.1, Math.min(4, nz)); img.style.width=(W*z)+'px'; img.style.height=(H*z)+'px'; ov.width=W*z; ov.height=H*z; render(); }
img.onload = () => { fit = Math.min((window.innerWidth-360)/W, (window.innerHeight-40)/H, 1); setZoom(fit); };

ov.addEventListener('mousedown', e => {
  const r = ov.getBoundingClientRect(), x=(e.clientX-r.left)/z, y=(e.clientY-r.top)/z;
  for(let i=0;i<14;i++){ if(pts[i]){ const dx=pts[i][0]-x, dy=pts[i][1]-y; if(dx*dx+dy*dy < (10/z)*(10/z)){ drag=i; active=i; render(); return; } } }
  pts[active]=[Math.round(x),Math.round(y)]; const nxt=pts.findIndex(p=>!p); active = nxt<0?active:nxt; render();
});
ov.addEventListener('mousemove', e => { if(drag<0) return; const r=ov.getBoundingClientRect(); pts[drag]=[Math.round((e.clientX-r.left)/z),Math.round((e.clientY-r.top)/z)]; render(); });
window.addEventListener('mouseup', ()=>{ drag=-1; });

function render(){
  ctx.clearRect(0,0,ov.width,ov.height);
  ctx.lineWidth=2; ctx.strokeStyle='rgba(80,200,120,0.85)';
  for(const [a,b] of EDGES){ if(pts[a]&&pts[b]){ ctx.beginPath(); ctx.moveTo(pts[a][0]*z,pts[a][1]*z); ctx.lineTo(pts[b][0]*z,pts[b][1]*z); ctx.stroke(); } }
  for(let i=0;i<14;i++){ if(!pts[i]) continue; const [x,y]=pts[i]; ctx.beginPath(); ctx.arc(x*z,y*z,6,0,7); ctx.fillStyle=i===active?'#fff':'#50C878'; ctx.fill(); ctx.fillStyle='#000'; ctx.font='bold 10px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,x*z,y*z); }
  const list=document.getElementById('list'); list.innerHTML='';
  LABELS.forEach((lab,i)=>{ const li=document.createElement('li'); li.className=(i===active?'active':'')+(pts[i]?' done':''); li.innerHTML='<span class="n">'+i+'</span>'+lab+(pts[i]?' ✓':''); li.onclick=()=>{active=i;render();}; list.appendChild(li); });
  document.getElementById('export').disabled = pts.some(p=>!p);
}
function doExport(){
  const blob=new Blob([JSON.stringify(pts)],{type:'application/json'}); const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='keypoints.json'; a.click(); URL.revokeObjectURL(a.href);
}
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Generate a standalone court-keypoint editor from a video frame")
    ap.add_argument("--video", required=True, help="path to the match video")
    ap.add_argument("--frame", type=int, default=None, help="frame index to grab (default: ~10%% in)")
    ap.add_argument("--out", default=None, help="output HTML path (default: alongside the video)")
    args = ap.parse_args()

    import cv2  # local import: keeps the module importable without OpenCV

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = args.frame if args.frame is not None else int(total * 0.1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {idx} from {args.video}")

    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    stem = Path(args.video).stem
    html = (
        _HTML_TEMPLATE
        .replace("__IMG__", b64)
        .replace("__LABELS__", json.dumps(KEYPOINT_LABELS))
        .replace("__EDGES__", json.dumps(EDGES))
        .replace("__WIDTH__", str(w))
        .replace("__HEIGHT__", str(h))
        .replace("__NAME__", stem)
    )
    out = Path(args.out) if args.out else Path(args.video).with_name(f"court_editor_{stem}.html")
    out.write_text(html)
    print(f"Frame {idx} of {total}  ({w}x{h})")
    print(f"Editor written: {out}")
    print(f"Open it in a browser, place the 14 points (order matters), download keypoints.json.")


if __name__ == "__main__":
    main()
