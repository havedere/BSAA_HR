"""Build the CV story deck from the notebook outputs (cache/deck/cv_story/*.png + values.json).
Writes cache/deck/cv_story/cv_story.fodp, then converts it with LibreOffice (flatpak) to
docs/cv_story.odp and cache/deck/cv_story/cv_story.pdf.
Layout and colors follow docs/hr_cnn_bilstm_results.odp (the two decks get merged).
Usage: .venv/bin/python .scratch/cv-story/gen_deck.py [--no-convert]"""
import base64
import json
import struct
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cache" / "deck" / "cv_story"
FODP = SRC / "cv_story.fodp"
V = json.loads((SRC / "values.json").read_text())

W, H = 33.867, 19.05
INK, MUTED, ACCENT, RED, CARD, DARK = "#212a30", "#7a7f84", "#1e7f5c", "#b03a2e", "#edf2f0", "#212a30"

# ---------- style registry ----------
_text_styles, _para_styles, _gr_styles = {}, {}, {}


def ts(size, color=INK, font="Calibri", bold=False):
    key = (size, color, font, bold)
    if key not in _text_styles:
        b = ' fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"' if bold else ""
        _text_styles[key] = (f"T{len(_text_styles) + 1}",
                             f'<style:text-properties fo:font-size="{size}pt" fo:color="{color}" style:font-name="{font}"{b}/>')
    return _text_styles[key][0]


def ps(align="start", below=0.2):
    key = (align, below)
    if key not in _para_styles:
        _para_styles[key] = (f"P{len(_para_styles) + 1}",
                             f'<style:paragraph-properties fo:text-align="{align}" fo:margin-top="0cm" fo:margin-bottom="{below}cm"/>')
    return _para_styles[key][0]


def gs(fill=None, valign="top"):
    key = (fill, valign)
    if key not in _gr_styles:
        f = f'draw:fill="solid" draw:fill-color="{fill}"' if fill else 'draw:fill="none"'
        _gr_styles[key] = (f"gr{len(_gr_styles) + 1}",
                           f'<style:graphic-properties {f} draw:stroke="none" draw:textarea-vertical-align="{valign}" '
                           'draw:auto-grow-height="false" draw:shadow="hidden" fo:padding-top="0cm" fo:padding-bottom="0cm" '
                           'fo:padding-left="0cm" fo:padding-right="0cm"/>')
    return _gr_styles[key][0]


# ---------- shapes ----------
def para(runs, align="start", below=0.2):
    """runs: str or list of (text, text-style)."""
    if isinstance(runs, tuple):
        runs = [runs]
    spans = "".join(f'<text:span text:style-name="{t}">{escape(s)}</text:span>' for s, t in runs)
    return f'<text:p text:style-name="{ps(align, below)}">{spans}</text:p>'


def box(x, y, w, h, paras, fill=None, valign="top", radius=None):
    rect = ""
    if fill:
        r = f' draw:corner-radius="{radius}cm"' if radius else ""
        rect = f'<draw:rect draw:style-name="{gs(fill)}"{r} svg:x="{x}cm" svg:y="{y}cm" svg:width="{w}cm" svg:height="{h}cm"/>'
        x, y, w, h = x + 0.5, y + 0.3, w - 1.0, h - 0.6
    return rect + (f'<draw:frame draw:style-name="{gs(None, valign)}" svg:x="{x}cm" svg:y="{y}cm" svg:width="{w}cm" '
                   f'svg:height="{h}cm"><draw:text-box>{"".join(paras)}</draw:text-box></draw:frame>')


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def image(name, x, y, max_w, max_h):
    """PNG fitted into the box (aspect kept, centred)."""
    path = SRC / f"{name}.png"
    pw, ph = png_size(path)
    s = min(max_w / pw, max_h / ph)
    w, h = pw * s, ph * s
    x, y = x + (max_w - w) / 2, y + (max_h - h) / 2
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<draw:frame draw:style-name="{gs()}" svg:x="{x:.3f}cm" svg:y="{y:.3f}cm" svg:width="{w:.3f}cm" '
            f'svg:height="{h:.3f}cm"><draw:image><office:binary-data>{data}</office:binary-data></draw:image></draw:frame>')


# ---------- slide templates ----------
slides = []
BODY = 16


def title_bar(title, kicker=None):
    return box(1.5, 1.2, W - 3, 2.2, [para((title, ts(32, INK, "Cambria", True)))], valign="middle")


def bullets(items, size=BODY):
    """items: str (bullet), or ('h', str) heading, or ('b', bold lead, rest)."""
    out = []
    for it in items:
        if isinstance(it, tuple) and it[0] == "h":
            out.append(para((it[1], ts(size + 1, INK, "Cambria", True)), below=0.15))
        elif isinstance(it, tuple) and it[0] == "b":
            out.append(para([("• ", ts(size)), (it[1], ts(size, INK, bold=True)), (it[2], ts(size))], below=0.3))
        else:
            out.append(para([("• ", ts(size)), (it, ts(size))], below=0.3))
    return out


def takeaway(text):
    return box(1.5, 16.9, W - 3, 1.6, [para([("Takeaway: ", ts(17, ACCENT, bold=True)), (text, ts(17))])])


def fig_slide(kicker, title, fig, items, take, fig_w=20.6):
    body = title_bar(title, kicker)
    body += image(fig, 1.5, 3.8, fig_w, 12.8)
    body += box(1.5 + fig_w + 0.6, 4.0, W - 3 - fig_w - 0.6, 12.6, bullets(items))
    body += takeaway(take)
    slides.append(("dpLight", body))


def wide_slide(kicker, title, fig, items, take, fig_h=9.6):
    body = title_bar(title, kicker)
    body += image(fig, 1.5, 3.6, W - 3, fig_h)
    body += box(1.5, 3.8 + fig_h, W - 3, 16.8 - 3.8 - fig_h, bullets(items, size=15))
    body += takeaway(take)
    slides.append(("dpLight", body))


def cards_slide(kicker, title, cards, take=None, top=4.0, ch=9.4):
    """cards: list of (heading, [items]) side by side."""
    body = title_bar(title, kicker)
    n, gap = len(cards), 0.6
    cw = (W - 3 - gap * (n - 1)) / n
    for i, (head, items) in enumerate(cards):
        paras = [para((head, ts(19, INK, "Cambria", True)), below=0.4)] + bullets(items, size=16)
        body += box(1.5 + i * (cw + gap), top, cw, ch, paras, fill=CARD, radius=0.35)
    if take:
        body += takeaway(take)
    slides.append(("dpLight", body))


f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"
pct = lambda x: f"{x:.0%}"
d, a, pr, b, c, fin, te, en, ga, sm, pi = (V[k] for k in ["data", "stage_a", "problem", "stage_b", "stage_c", "final",
                                                         "test", "ensemble", "gate", "smooth", "pipeline"])
sm_d = lambda p, m: sm[f"dist_bootHL_{p}_{m}"]

# ---------- 1 title ----------
slides.append(("dpDark",
    box(2.0, 6.0, W - 4, 4.5, [para(("Heart Rate from Wearable PPG:", ts(40, "#ffffff", "Cambria", True)), "center", 0),
                               para(("What 135 Cross-validated Runs Show", ts(40, "#ffffff", "Cambria", True)), "center", 0)],
        valign="bottom")
    + box(2.0, 11.0, W - 4, 1.2, [para(("Model selection (Stages A-C), seed ensembles, quality gate and smoothing (Stages D-F)",
                                        ts(20, "#c9d1cd")), "center", 0)])
    + box(2.0, 17.8, W - 2.6, 0.8, [para(("Julian Wollner, Thomas Schwab, Bedran Aksu, Patricia Glavas", ts(12, "#c9d1cd")),
                                         "end", 0)])))

# ---------- 2 the question ----------
cards_slide("Overview", "The question, in three acts", [
    ("1  Choose the model", ["9 training setups x 5 folds x 3 seeds = 135 runs",
                             "Stage A: loss · Stage B: bootstrap + LDS · Stage C: distribution head",
                             "Decided on validation only, checked per test fold"]),
    ("2  The problem it reveals", ["Every model pulls predictions toward the middle (75-85 BPM)",
                                   "High heart rates are under-predicted by tens of BPM",
                                   "Root cause: the label distribution"]),
    ("3  Make the output usable", ["Stage D: average 3 seeds", "Stage E: drop windows the model is unsure about",
                                   "Stage F: smooth along time (Kalman, HMM)"]),
], take="Input: one 10 s PPG window (100 Hz, band-pass 0.5-8 Hz). Output: heart rate in BPM. Truth: ECG R-peaks.")

# ---------- 3 data ----------
fig_slide("Data", "Two datasets, one crowded middle", "data_labels", [
    ("b", f"{d['subjects']} subjects", f": {d['subjects_multisite-ppg']} multisite-PPG (free-living, 4 wearable sites), "
                                        f"{d['subjects_wrist-finger-ppg']} wrist-finger-PPG (short controlled clips)"),
    ("b", f"{d['windows']:,} windows", " of 10 s, 2 s stride, at most 2000 per subject"),
    ("b", f"{d['share_60_90']:.0f} %", " of all windows have a label of 60-90 BPM"),
    ("b", f"{d['share_110plus']:.1f} %", " are at 110 BPM or more, almost all from multisite"),
], take="A model that minimises average error can mostly ignore heart rates above 110 BPM. It does, as we will see.")

# ---------- 4 protocol ----------
cards_slide("Protocol", "Selection protocol: CV, seeds, decision rule", [
    ("5-fold subject CV", ["5 subject groups per dataset (stratified)", "Fold k: test = group k, val = group k+1, train = rest",
                           "Every subject is tested exactly once", "No subject in two splits"]),
    ("3 seeds", ["Seeds 42, 1, 2: init, shuffle, augmentation", "Per subject: median MAE over seeds",
                 "Same split for every seed"]),
    ("Decision rule", ["Key: per-subject val MAE gap to the best setup (paired), averaged per dataset",
                       "Tie set: within 1 SE (subject bootstrap)", "Tie-break: bin-balanced val MAE (tails)",
                       "LTGO check: re-decide without the val set of each test group"]),
], take="Test results are computed after the choice and never feed back into it.")

# ---------- 5 stage A ----------
body = title_bar("Stage A: MSE wins, by a small margin", "Stage A · loss")
body += image("stage_a_keys", 1.5, 3.1, 20.6, 5.6)
body += image("stage_a_folds", 1.5, 8.9, 20.6, 7.7)
body += box(22.7, 3.4, W - 24.2, 13.2, bullets([
    ("b", "MSE wins", f" globally and in all {a['ltgo_agree']} LTGO checks"),
    f"Huber δ10 +{f2(a['key_huber10'])} BPM (SE {f2(a['se_huber10'])}), MAE +{f2(a['key_mae'])} (SE {f2(a['se_mae'])})",
    "Huber δ10 is inside the tie set in 3 of 5 LTGO checks",
    ("b", "Folds matter more than losses:", " per-fold val MAE spans ~4.1-5.3 BPM, losses differ by 0.1-0.3"),
]))
body += takeaway("L* = MSE. The choice of loss is a minor lever.")
slides.append(("dpLight", body))

# ---------- 6 the problem ----------
fig_slide("The problem", "All models pull toward the middle", "problem", [
    ("b", f"Slope {f2(pr['slope'])}:", f" a 40 BPM rise in true heart rate moves the prediction by only {40 * pr['slope']:.0f} BPM"),
    ("b", f"≥ 120 BPM:", f" under-predicted by {abs(pr['bias_120plus']):.0f} BPM on average (MAE {f1(pr['mae_120plus'])}, "
                         f"n = {pr['n_120plus']:,} windows)"),
    ("b", "40-60 BPM:", f" over-predicted by {f1(pr['bias_40_60'])} BPM"),
    f"70-80 BPM: MAE {f1(pr['mae_70_80'])}, where most data lives",
    "MSE, test windows of all 5 folds and 3 seeds",
], take="Average MAE hides it: the model is good where data is dense and close to useless at high heart rates.", fig_w=21.5)

# ---------- 7 stage B ----------
fig_slide("Stage B · bootstrap + LDS", "Stage B: bootstrap and LDS trade, not win", "stage_b_bins", [
    ("b", "Bootstrap HL:", " time-warped extra windows in the low and high bins"),
    ("b", "LDS p:", " loss weight ∝ (smoothed label density)^-p; p = 0 is off"),
    f"LDS 1 cuts 130+ BPM error from 47 to 31, but the 70-90 middle rises from ~3 to ~8",
    ("b", f"Winner: MSE + boot (p = 0)", f", in all {b['ltgo_agree']} LTGO checks; p = 0.5 is +{f2(b['key_mse_bootHL_lds0.5'])} BPM"),
    f"Still {f2(fin['all_key_mse_bootHL'])} BPM behind plain MSE overall",
], take="With a regression head, every BPM gained in the tails is paid for in the middle.")

# ---------- 8 stage C ----------
fig_slide("Stage C · distribution head", "Stage C: the distribution head wins", "tradeoff", [
    ("b", "dist head:", " 191 bins (30-220 BPM), Gaussian soft targets, BPM = expected value"),
    ("b", "dist + boot is best overall:", f" MSE is +{f2(fin['all_key_mse'])} BPM (SE {f2(fin['all_se_mse'])}), "
                                          f"MSE + boot +{f2(fin['gap_B_vs_C'])} (SE {f2(fin['se_B_vs_C'])})"),
    f"…and better in the tails than MSE: bin-balanced {f1(fin['all_binbal_dist_bootHL'])} vs {f1(fin['all_binbal_mse'])} BPM",
    "LDS shows the same trade as in Stage B: further down (tails), further right (overall)",
    ("b", "Final setup: dist + boot", f", chosen in all {fin['ltgo_agree']} LTGO checks"),
], take="Lower-left is better. Only dist + boot improves both axes over MSE; LDS moves along the trade-off.")

# ---------- 9 bias of winners ----------
fig_slide("The problem, revisited", "The winners shrink the pull, but do not remove it", "bias_winners", [
    ("b", "Slope", f" {f2(pr['slope_mse'])} (MSE) → {f2(pr['slope_mse_bootHL'])} (MSE + boot) → {f2(pr['slope_dist_bootHL'])} (dist + boot)"),
    ("b", "≥ 120 BPM MAE", f" {f1(pr['mae_120plus_mse'])} → {f1(pr['mae_120plus_mse_bootHL'])} → {f1(pr['mae_120plus_dist_bootHL'])}"),
    "Low bins improve too: 40-60 BPM bias +16 → +13",
    "A slope of 1 would mean no pull at all",
], take="The pull to the middle is the open problem of this model: the best setup still halves high-BPM changes.")

# ---------- 10 test ----------
fig_slide("Test results", "Test: dist + boot ahead, by a small margin", "test_oof", [
    ("b", f"dist + boot {f2(te['oof_dist_bootHL'])} BPM", f" vs MSE {f2(te['oof_mse'])} (out-of-fold, mean of 3 seeds)"),
    f"Difference to MSE: 95 % CI [{f2(te['diff_lo_dist_bootHL'])}, {f2(te['diff_hi_dist_bootHL'])}] BPM",
    f"Wilcoxon (Holm): wrist-finger p = {te['p_wf_dist_bootHL']:.3f}, multisite p = {te['p_ms_dist_bootHL']:.2f}",
    "So: the gain is clear on wrist-finger, not shown on multisite (20 subjects)",
    "The test ranking matches the validation ranking at the top",
], take="Validation choice holds on test. The improvement is real but about 0.3 BPM, not a step change.")

# ---------- 11 ensemble ----------
fig_slide("Stage D · seed ensemble", "Stage D: three seeds, a free 0.2 BPM", "ensemble", [
    "Reg head: mean of the 3 predictions; dist head: mean of the 3 softmax curves",
    ("b", "dist + boot:", f" {f2(en['single_dist_bootHL'])} → {f2(en['ens_dist_bootHL'])} BPM"),
    f"MSE: {f2(en['single_mse'])} → {f2(en['ens_mse'])}; MSE + boot: {f2(en['single_mse_bootHL'])} → {f2(en['ens_mse_bootHL'])}",
    "Same gain for every setup; the ranking does not change",
    "The seed spread also becomes an uncertainty score (next slide)",
], take="The three seeds were trained anyway; the ensemble costs only inference time.")

# ---------- 12 gate ----------
g = lambda cfg, s, m: ga[f"{cfg}_{s}_{m}"]
fig_slide("Stage E · quality gate", "Stage E: which windows to trust", "gate_curves", [
    "Limit = 90 % quantile of the score on val, applied to test",
    ("b", "Curve std (dist):", f" 90 % kept, MAE {f2(g('dist_bootHL', 'score_std', 'test_mae_all'))} → "
                               f"{f2(g('dist_bootHL', 'score_std', 'test_mae_passed'))} BPM, ρ = {f2(g('dist_bootHL', 'score_std', 'test_rho'))}"),
    f"AURC: curve std {f2(g('dist_bootHL', 'score_std', 'test_aurc'))}, MC dropout {f2(g('dist_bootHL', 'score_mc', 'test_aurc'))}, "
    f"seed spread {f2(g('dist_bootHL', 'score_spread', 'test_aurc'))}",
    f"MC dropout barely ranks errors for the reg head (ρ = {f2(g('mse', 'score_mc', 'test_rho'))})",
    "PPG-only scores (skew, prominence) are weaker than any model score",
], take="The distribution head gives an uncertainty for free, and it is the best gate we have.")

# ---------- 13 gate cost ----------
cov = lambda *bins: [ga[f"cov_bin_score_std_cov_{b}"] for b in bins]
fig_slide("Stage E · the cost", "Stage E: the gate drops high heart rates", "gate_bins", [
    "Coverage per BPM bin for dist + boot at the 90 % limit",
    ("b", "60-90 BPM:", f" {pct(min(cov('60', '70', '80')))}-{pct(max(cov('60', '70', '80')))} kept"),
    ("b", "≥ 110 BPM:", f" only {pct(min(cov('110', '120', '130')))}-{pct(max(cov('110', '120', '130')))} kept"),
    "Good: the model knows when it is unsure",
    "But: part of the MAE gain comes from not answering at high heart rates",
], take="Always report coverage per BPM range next to the gated MAE, or the gate hides the tail problem.")

# ---------- 14 smoothing example ----------
wide_slide("Stage F · smoothing", "Stage F: smoothing one test recording", "smooth_example", [
    "Every 2 s window predicted; Kalman + RTS uses the window's uncertainty as measurement noise, input gate gives "
    "unsure windows no weight, output gate drops windows whose smoothed std stays high (red)",
    f"{sm['example']}: raw MAE {f1(sm['example_raw'])} → {f1(sm['example_kalman'])} BPM on the kept windows",
], take="Artefact bursts (40-48 min) produce wild single-window predictions; gate + smoother ride through them.", fig_h=10.6)

# ---------- 15 smoothing results ----------
fig_slide("Stage F · results", "Stage F: Kalman or HMM + gates ≈ 3.1 BPM", "smooth_pareto", [
    ("b", "dist + boot, no gate:", f" raw {f2(sm_d('raw', 'mae'))} → median {f2(sm_d('median', 'mae'))} → "
                                  f"Kalman {f2(sm_d('Kalman', 'mae'))} / HMM {f2(sm_d('HMM', 'mae'))}"),
    ("b", "With gates:", f" Kalman {f2(sm_d('Kalman_plus_gates', 'mae'))} at {pct(sm_d('Kalman_plus_gates', 'coverage'))}, "
                         f"HMM {f2(sm_d('HMM_plus_gates', 'mae'))} at {pct(sm_d('HMM_plus_gates', 'coverage'))}"),
    f"Gate without smoother: {f2(sm_d('raw_plus_in_gate', 'mae'))} at {pct(sm_d('raw_plus_in_gate', 'coverage'))}",
    f"MSE best pipeline: {f2(sm['mse_Kalman_plus_gates_mae'])} BPM",
    f"Error after the largest HR jump: {f1(sm_d('Kalman_plus_gates', 'lag_mae'))} (dist) vs {f1(sm['mse_Kalman_plus_gates_lag_mae'])} (MSE) BPM",
], take="Smoothing and gating help every setup; dist + boot stays about 0.9 BPM ahead of MSE.")

# ---------- 16 pipeline ----------
fig_slide("Summary", "From one model to the full pipeline", "pipeline_steps", [
    ("b", f"{f2(pi['single model'])} → {f2(pi['+ in/out gate'])} BPM", f" at {pct(pi['final_coverage'])} coverage"),
    f"Ensemble −{f2(pi['single model'] - pi['3-seed ensemble'])}, Kalman −{f2(pi['ensemble, all windows'] - pi['+ Kalman smoothing'])}, "
    f"gates −{f2(pi['+ Kalman smoothing'] - pi['+ in/out gate'])}",
    "Bars 1-2: capped test windows (≤ 2000 per subject); bars 3-5: every labelled window of the recordings",
    "Mean over the 5 test folds, subject-macro MAE",
], take="Final pipeline: dist head + time-warp bootstrap, 3 seeds, Kalman + gates → 3.1 BPM on 86 % of windows.")

# ---------- 17 limitations ----------
cards_slide("Limitations", "What these numbers do not show", [
    ("Selection", ["Greedy stages, no full grid: e.g. Huber + bootstrap or dist without bootstrap never ran",
                   "Decision on overall MAE; tails only break ties", "3 seeds; validation MAE moves ±1 BPM between epochs"]),
    ("Data", ["High heart rates come almost only from multisite (20 subjects)", "Wrist-finger has no labels above 115 BPM",
              "Exercise dataset excluded"]),
    ("Gate and smoothing", ["Tuned per fold on val, scored on test, 5-fold mean", "Gated MAE is on fewer windows, fewer at high BPM",
                            "Lag error uses one jump per recording"]),
])

# ---------- 18 conclusion ----------
cards_slide("Conclusion", "Takeaways", [
    ("What works", [f"Distribution head + time-warp bootstrap: best single model ({f2(te['oof_dist_bootHL'])} BPM test)",
                    "Its curve width is the best quality score",
                    f"Ensemble + Kalman + gates: {f2(pi['+ in/out gate'])} BPM at {pct(pi['final_coverage'])} coverage"]),
    ("What does not", ["The loss choice (≤ 0.3 BPM)", "LDS: trades middle accuracy for tail accuracy, never wins overall"]),
    ("Open problem", [f"Pull to the middle: slope {f2(pr['slope_dist_bootHL'])}, ≥ 120 BPM MAE {f1(pr['mae_120plus_dist_bootHL'])}",
                      "Losses and weights did not fix it; more high-heart-rate data is the likely lever"]),
], take=f"dist + boot, 3 seeds, Kalman + gates: {f2(pi['+ in/out gate'])} BPM on {pct(pi['final_coverage'])} of windows; "
        "high heart rates remain the weak spot.")

# ---------- write ----------
pages = "".join(f'<draw:page draw:name="Slide{i + 1}" draw:style-name="{bg}" draw:master-page-name="Default">{body}</draw:page>'
                for i, (bg, body) in enumerate(slides))
auto = (f'<style:page-layout style:name="PM1"><style:page-layout-properties fo:page-width="{W}cm" fo:page-height="{H}cm" '
        'fo:margin-top="0cm" fo:margin-bottom="0cm" fo:margin-left="0cm" fo:margin-right="0cm" style:print-orientation="landscape"/></style:page-layout>'
        '<style:style style:name="dpLight" style:family="drawing-page"><style:drawing-page-properties draw:fill="solid" draw:fill-color="#ffffff" '
        'presentation:background-visible="true" presentation:background-objects-visible="true"/></style:style>'
        f'<style:style style:name="dpDark" style:family="drawing-page"><style:drawing-page-properties draw:fill="solid" draw:fill-color="{DARK}" '
        'presentation:background-visible="true" presentation:background-objects-visible="true"/></style:style>'
        + "".join(f'<style:style style:name="{n}" style:family="graphic">{x}</style:style>' for n, x in _gr_styles.values())
        + "".join(f'<style:style style:name="{n}" style:family="paragraph">{x}</style:style>' for n, x in _para_styles.values())
        + "".join(f'<style:style style:name="{n}" style:family="text">{x}</style:style>' for n, x in _text_styles.values()))
ns = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
      'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
      'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" xmlns:xlink="http://www.w3.org/1999/xlink"')
doc = (f'<?xml version="1.0" encoding="UTF-8"?>\n<office:document {ns} office:version="1.3" '
       'office:mimetype="application/vnd.oasis.opendocument.presentation">'
       '<office:font-face-decls><style:font-face style:name="Cambria" svg:font-family="Cambria"/>'
       '<style:font-face style:name="Calibri" svg:font-family="Calibri"/></office:font-face-decls>'
       '<office:styles><style:default-style style:family="graphic"><style:text-properties style:font-name="Calibri" '
       'fo:font-size="14pt"/></style:default-style></office:styles>'
       f'<office:automatic-styles>{auto}</office:automatic-styles>'
       '<office:master-styles><style:master-page style:name="Default" style:page-layout-name="PM1" draw:style-name="dpLight"/></office:master-styles>'
       f'<office:body><office:presentation>{pages}</office:presentation></office:body></office:document>')
FODP.write_text(doc)
print(f"wrote {FODP} ({len(slides)} slides)")

if "--no-convert" not in sys.argv:
    lo = ["flatpak", "run", "org.libreoffice.LibreOffice", "--headless", "--convert-to"]
    subprocess.run([*lo, "odp", "--outdir", str(ROOT / "docs"), str(FODP)], check=True)
    subprocess.run([*lo, "pdf", "--outdir", str(SRC), str(FODP)], check=True)
    print(f"wrote {ROOT / 'docs' / 'cv_story.odp'}, {SRC / 'cv_story.pdf'}")
