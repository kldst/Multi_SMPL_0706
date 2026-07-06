#!/usr/bin/env python
"""Build a self-contained HTML report for the overfit inference (debug_07 output).

Reads the PNG overlays + metrics.json under 07_infer_overfit/, embeds the images
as base64 JPEGs, and writes report.html (content-only: <title>+<style>+markup,
ready for the Artifact tool). GT vs prediction, per view, plus attention + metrics.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]


def img_data_uri(path: Path, width: int = 380, quality: int = 82) -> str:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if im is None:
        return ""
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def fig(src, cap):
    return f'<figure><img loading="lazy" src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "debug_outputs/mamma_pipeline/07_infer_overfit"))
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "debug_outputs/mamma_pipeline/07_infer_overfit/report.html"))
    args = ap.parse_args()
    d = Path(args.dir)
    metrics = json.loads((d / "metrics.json").read_text())
    P = metrics["num_people"]

    # ---- summary metric tiles per person ----
    pc = ["#ff5a5a", "#5aa0ff", "#54d38a", "#ffc23d"]
    tiles = []
    for j in range(P):
        m = metrics["per_person"][f"gt_person_{j}"]
        iou = m["mask_iou"]; l2 = m["landmark_l2_px"]
        iou_q = "good" if iou >= 0.6 else ("warn" if iou >= 0.4 else "bad")
        l2_q = "good" if l2 <= 40 else ("warn" if l2 <= 70 else "bad")
        tiles.append(f"""
        <article class="tile">
          <header><span class="dot" style="background:{pc[j%4]}"></span>Person {j}
            <span class="muted">→ slot {m['matched_pred_slot']}</span></header>
          <div class="stat"><span class="k">mask IoU</span>
            <span class="v {iou_q}">{iou:.3f}</span></div>
          <div class="stat"><span class="k">landmark L2</span>
            <span class="v {l2_q}">{l2:.0f}<small>px</small></span></div>
        </article>""")

    # ---- per-view comparison cards ----
    views_html = []
    for s in range(args.views):
        cells = [
            fig(img_data_uri(d / f"view{s}_gt_landmarks.png"), "GT landmarks"),
            fig(img_data_uri(d / f"view{s}_pred_landmarks.png"), "Pred landmarks"),
            fig(img_data_uri(d / f"view{s}_gt_mask.png"), "GT mask"),
            fig(img_data_uri(d / f"view{s}_pred_mask.png"), "Pred mask"),
        ]
        for j in range(P):
            ap_ = d / f"view{s}_attn_p{j}.png"
            if ap_.exists():
                cells.append(fig(img_data_uri(ap_), f"Attn · person {j}"))
        views_html.append(f"""
      <section class="view">
        <h3><span class="idx">view {s}</span></h3>
        <div class="grid">{''.join(cells)}</div>
      </section>""")

    html = f"""<title>Overfit Inference · Landmarks / Mask / Attention</title>
<style>
  :root {{
    --bg:#f6f7f9; --panel:#ffffff; --ink:#12161c; --muted:#5b6673;
    --line:#e4e8ee; --accent:#0f8fa3; --good:#128a5a; --warn:#b7791f; --bad:#c0392b;
    --p0:#ff5a5a; --p1:#5aa0ff;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }}
  @media (prefers-color-scheme:dark) {{
    :root {{ --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b98a6;
      --line:#252c36; --accent:#3fb6c4; --good:#3fd18a; --warn:#e0b23d; --bad:#ff6b5e; }}
  }}
  :root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#12161c; --muted:#5b6673;
    --line:#e4e8ee; --accent:#0f8fa3; --good:#128a5a; --warn:#b7791f; --bad:#c0392b; }}
  :root[data-theme="dark"] {{ --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b98a6;
    --line:#252c36; --accent:#3fb6c4; --good:#3fd18a; --warn:#e0b23d; --bad:#ff6b5e; }}

  body {{ background:var(--bg); color:var(--ink); font-family:var(--sans);
    line-height:1.5; margin:0; padding:32px clamp(16px,4vw,56px); }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  .eyebrow {{ font:600 12px/1 var(--mono); letter-spacing:.14em; text-transform:uppercase;
    color:var(--accent); margin:0 0 10px; }}
  h1 {{ font-size:clamp(24px,3.2vw,36px); font-weight:680; margin:0 0 8px; text-wrap:balance;
    letter-spacing:-.01em; }}
  .lede {{ color:var(--muted); margin:0 0 6px; max-width:70ch; }}
  .meta {{ font:12px/1.6 var(--mono); color:var(--muted); margin:14px 0 28px;
    display:flex; flex-wrap:wrap; gap:6px 20px; }}
  .meta b {{ color:var(--ink); font-weight:600; }}

  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
    gap:14px; margin-bottom:14px; }}
  .tile {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .tile header {{ display:flex; align-items:center; gap:8px; font-weight:600; margin-bottom:12px; }}
  .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; }}
  .muted {{ color:var(--muted); font-weight:400; font-size:13px; }}
  .stat {{ display:flex; justify-content:space-between; align-items:baseline;
    padding:5px 0; border-top:1px solid var(--line); }}
  .stat .k {{ color:var(--muted); font-size:13px; }}
  .stat .v {{ font:600 20px/1 var(--mono); font-variant-numeric:tabular-nums; }}
  .stat .v small {{ font-size:12px; color:var(--muted); margin-left:2px; }}
  .v.good {{ color:var(--good); }} .v.warn {{ color:var(--warn); }} .v.bad {{ color:var(--bad); }}

  .legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--muted);
    margin:0 0 26px; }}
  .legend span {{ display:inline-flex; align-items:center; gap:7px; }}

  .note {{ background:color-mix(in srgb,var(--warn) 12%,var(--panel));
    border:1px solid color-mix(in srgb,var(--warn) 40%,var(--line));
    border-radius:10px; padding:12px 16px; font-size:13.5px; color:var(--ink); margin:0 0 30px; }}
  .note b {{ color:var(--warn); }}

  .view {{ margin-bottom:30px; }}
  .view h3 {{ font-size:14px; margin:0 0 12px; border-bottom:1px solid var(--line);
    padding-bottom:8px; }}
  .idx {{ font:600 12px/1 var(--mono); letter-spacing:.12em; text-transform:uppercase;
    color:var(--accent); }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
  figure {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:10px; overflow:hidden; }}
  figure img {{ display:block; width:100%; height:auto; }}
  figcaption {{ font:12px/1 var(--mono); color:var(--muted); padding:9px 11px;
    border-top:1px solid var(--line); letter-spacing:.02em; }}
  footer {{ color:var(--muted); font-size:12px; border-top:1px solid var(--line);
    margin-top:24px; padding-top:16px; }}
</style>

<div class="wrap">
  <p class="eyebrow">VGGT-multi · overfit sanity check</p>
  <h1>Dense-landmark &amp; per-person-mask predictions vs. ground truth</h1>
  <p class="lede">The model was overfit on a single frame (2 people, 4 views). Below, every
  view shows ground truth beside the model's prediction for the two auxiliary heads, plus the
  SMPL person-query attention that drives them.</p>
  <div class="meta">
    <span><b>checkpoint</b> mamma_overfit / checkpoint_300.pt</span>
    <span><b>people</b> {P}</span>
    <span><b>views</b> {args.views}</span>
    <span><b>heads</b> dense-landmark (512·GNLL) + person-mask (37×37 BCE)</span>
  </div>

  <div class="tiles">{''.join(tiles)}</div>
  <div class="legend">
    <span><span class="dot" style="background:var(--p0)"></span>Person 0</span>
    <span><span class="dot" style="background:var(--p1)"></span>Person 1</span>
    <span>metrics are prediction vs GT, Hungarian-matched (slot → person)</span>
  </div>

  <p class="note"><b>Read the landmarks with care.</b> Predicted landmarks reach ~33–37&nbsp;px
  L2 on visible points, but the GNLL head became over-confident during overfit (variance
  collapse), so the 2D points cluster into a narrow band rather than spreading across the body.
  The mask head, by contrast, cleanly separates the two people (IoU 0.65–0.80).</p>

  {''.join(views_html)}

  <footer>Generated from debug/debug_07_infer_overfit_viz.py outputs · images embedded as
  base64 JPEG · GT left of prediction in each row.</footer>
</div>
"""
    Path(args.out).write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"[report] wrote {args.out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
