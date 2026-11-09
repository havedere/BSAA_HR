"""Replace the 'Results' placeholder slide of the deck with the final OOF result slides.
Usage: python add_results.py <unzipped deck dir> <plots dir>"""
import re
import shutil
import sys
from pathlib import Path

X, PLOTS = Path(sys.argv[1]), Path(sys.argv[2])
content = (X / "content.xml").read_text()
manifest = (X / "META-INF" / "manifest.xml").read_text()

IMG_W = {"bland_altman": (2000, 924), "scatter": (2000, 1134), "bins": (2000, 1344),
         "subjects": (2000, 756), "error_cdf": (2000, 714),
         "train_test_folds": (2000, 798), "train_test_subjects": (2000, 840)}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def title(t):
    return ('<draw:frame draw:style-name="gr1" draw:text-style-name="P3" draw:layer="layout" svg:width="30.867cm" '
            f'svg:height="2.2cm" svg:x="1.5cm" svg:y="1.2cm"><draw:text-box><text:p text:style-name="P6">'
            f'<text:span text:style-name="T4">{esc(t)}</text:span></text:p></draw:text-box></draw:frame>')


def image(name, x, y, w):
    pw, ph = IMG_W[name]
    href = f"Pictures/final_{name}.png"
    return (f'<draw:frame draw:style-name="gr10" draw:text-style-name="P3" draw:layer="layout" svg:width="{w:.3f}cm" '
            f'svg:height="{w * ph / pw:.3f}cm" svg:x="{x:.3f}cm" svg:y="{y:.3f}cm"><draw:image xlink:href="{href}" '
            'xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad" draw:mime-type="image/png"><text:p/>'
            '</draw:image></draw:frame>')


def bullets(items, x=23.6, w=8.767):
    ps = "".join(f'<text:p text:style-name="P30"><text:span text:style-name="T39">• </text:span>'
                 f'<text:span text:style-name="T40">{esc(b)}</text:span>'
                 f'<text:span text:style-name="T39"> {esc(r)}</text:span></text:p>' for b, r in items)
    return (f'<draw:frame draw:style-name="gr2" draw:text-style-name="P3" draw:layer="layout" svg:width="{w}cm" '
            f'svg:height="12.6cm" svg:x="{x}cm" svg:y="4cm"><draw:text-box>{ps}</draw:text-box></draw:frame>')


def takeaway(t, label="Takeaway: "):
    return ('<draw:frame draw:style-name="gr2" draw:text-style-name="P3" draw:layer="layout" svg:width="30.867cm" '
            'svg:height="1.6cm" svg:x="1.5cm" svg:y="16.9cm"><draw:text-box><text:p text:style-name="P6">'
            f'<text:span text:style-name="T14">{esc(label)}</text:span><text:span text:style-name="T9">{esc(t)}</text:span>'
            '</text:p></draw:text-box></draw:frame>')


def notes(n, text):
    return ('<presentation:notes draw:style-name="dp2"><draw:page-thumbnail draw:style-name="gr3" draw:layer="layout" '
            'svg:width="18.624cm" svg:height="10.476cm" svg:x="1.482cm" svg:y="2.123cm" '
            f'draw:page-number="{n}" presentation:class="page"/><draw:frame presentation:style-name="pr1" '
            'draw:text-style-name="P5" draw:layer="layout" svg:width="17.271cm" svg:height="12.572cm" svg:x="2.159cm" '
            f'svg:y="13.271cm" presentation:class="notes"><draw:text-box>' + ''.join(f'<text:p>{esc(t)}</text:p>' for t in ([text] if isinstance(text, str) else text)) + '</draw:text-box>'
            '</draw:frame></presentation:notes>')


def page(name, n, body, note):
    return (f'<draw:page draw:name="{name}" draw:style-name="dp3" draw:master-page-name="Default">{body}'
            f'{notes(n, NOTES.get(name, note))}</draw:page>')


CAP = ("Windows: the cached labelled windows of cache/windows/, i.e. the set training sampled from "
       "(multisite capped at 2,000 random windows per subject, wrist-finger complete); no bootstrap windows, no noise.")
NOTES = {
    "Results": [
        "What: headline numbers of the winner pipeline (dist head + bootstrap HL) on unseen subjects.",
        "How: subject-wise 5-fold CV. Each subject is tested exactly once, by the 3-seed ensemble (seeds 42, 1, 2) of "
        "the fold whose test group it is in; those 3 models never saw the subject. Gate limit and Kalman parameters "
        "(m, c) were tuned on that fold's val group, also other subjects.",
        "Rows: raw ensemble = average of the 3 softmax curves, decoded to BPM, no gate or smoothing. Shipped pipeline, "
        "every window = input gate + Kalman + RTS, no output gate; this is what estimate_heartrate returns. Gated gaps "
        "<= 30 s only = the scope of smooth_report.py, which leaves longer gated runs out (98.5 % of windows remain).",
        "Metric: MAE per subject, averaged per sub-dataset, then the mean of the 2 sub-datasets (so the 414k multisite "
        "windows do not dominate). Windows: every labelled 10 s window (2 s step) of the test recordings, 428k, not capped.",
        "The submitted model is the 15-model ensemble (all folds). It has no test data left, so these CV numbers are "
        "its performance estimate (slightly pessimistic: each fold model saw only 3/5 of the subjects).",
        "Source: notebooks/final_results.ipynb, cache/results/final_dist_bootHL.csv.",
    ],
    "ResultsBA": [
        "What: Bland-Altman of the shipped pipeline, all out-of-fold test windows, one panel per sub-dataset.",
        "Axes: x = mean of prediction and ECG truth, y = prediction minus truth. Hexagons count windows, color on a "
        "log scale (dark = many).",
        "Lines: solid = bias (mean difference); dashed = 95 % limits of agreement, bias +- 1.96 SD of the differences.",
        "Read: wrist-finger is a tight band around 0 (limits about +-4.4 BPM). Multisite is unbiased on average "
        "(+1.2) but wide (about +-21 BPM). The falling diagonal in multisite comes from windows where the prediction "
        "stays near 80 BPM while the truth moves: then difference and mean change together.",
        "Caveat: windows overlap (10 s every 2 s), so the points are not independent; the limits describe the spread "
        "of the error, they are not a confidence interval.",
    ],
    "ResultsScatter": [
        "What: predicted BPM against ECG truth, same windows as the Bland-Altman slide, density on a log scale.",
        "Lines: grey = perfect agreement (y = x); red dashed = least-squares fit. r = Pearson correlation over windows.",
        "Read: a fit slope below 1 means the prediction is pulled toward the middle. Wrist-finger: slope 0.93, r 0.98. "
        "Multisite: slope 0.49, r 0.72, with a horizontal band at 80-90 BPM: where the PPG shows no clear pulse the "
        "model falls back to the typical heart rate of the training data.",
        "Gate + Kalman raise multisite r from 0.57 (raw ensemble) to 0.72.",
    ],
    "ResultsBins": [
        "What: error per 10-BPM bin of the ECG truth, all out-of-fold test windows, both sub-datasets pooled.",
        "Top: MAE per bin; grey = raw ensemble, green = shipped pipeline (Kalman + input gate). The number above each "
        "bar pair is the window count of the bin.",
        "Bottom: bias of the shipped pipeline (prediction minus truth). Positive at low heart rates, negative at high "
        "ones: the pull toward the middle.",
        "Read: best at 80-90 BPM (MAE 2.2) where most data is; 130+ BPM is under-predicted by about 42 BPM but holds only "
        "2,820 windows (0.7 %). Smoothing lowers MAE in every bin by about 1 BPM but cannot remove the bias.",
    ],
    "ResultsSubjects": [
        "Top: MAE of each test subject over all its windows, shipped pipeline, sorted, colored by sub-dataset. Every "
        "subject appears once, scored by the fold ensemble that never saw it.",
        "Bottom: cumulative error distribution. For each error x on the horizontal axis, the share of windows with "
        "|prediction - truth| <= x. Dashed = raw ensemble, solid = shipped pipeline. A curve further up/left is better; "
        "the dotted lines mark 5 and 10 BPM.",
        "Read: every wrist-finger subject is at or below 3.4 BPM (median 1.14); multisite subjects spread from about 2 "
        "to 12.4 BPM (median 5.30), so the multisite error is not one outlier. Within 5 BPM: 96 % of wrist-finger, 63 % "
        "of multisite windows.",
    ],
    "TrainTestFolds": [
        "What: train vs val vs test MAE for each of the 5 fold ensembles (raw 3-seed ensemble, no gate or smoothing).",
        "How: fold k's ensemble is scored on its own train subjects (3 groups, trained on them), its val group "
        "(group k+1, used for early stopping) and its test group (group k, never used). Subject-macro MAE per "
        "sub-dataset. " + CAP,
        "Last row: mean +- sd over the 5 folds. Both sub-datasets: train 3.39 +- 0.27, val 4.00 +- 0.37, test 4.18 +- 0.27 BPM.",
        "How to read: over-fitting is the gap between train (left) and the unseen splits (val, test). Val and test are "
        "both unseen subjects, so they should sit close together; val is slightly optimistic because it chose the "
        "stopping epoch. Strong over-fitting would be train near 0 with val and test far right; all three high and "
        "close = the data limits the model (multisite partly); all low and close = good fit (wrist-finger).",
        "Single folds jump (e.g. wrist-finger fold 2 test left of train) because each group holds only about 4 "
        "multisite and 5-6 wrist-finger subjects; read the mean row.",
        "Test 4.18 here vs 4.11 raw on the results slide: different window sets (cached/capped here, all labelled "
        "windows there).",
    ],
    "TrainTestSubjects": [
        "What: per-subject MAE in train, val and test, raw 3-seed ensembles. One dot = one subject; thin lines join the "
        "same subject across the three boxes.",
        "Which model: every value comes from a 3-seed ensemble, but not the same one. Test: the fold where the subject "
        "is in test (out of fold, never seen). Val: the fold where it is val (not trained on, used for early stopping). "
        "Train: mean over the 3 folds where the subject was in train (trained on it).",
        "So a line compares how the recipe does on a subject it trained on vs the same subject when it never saw it.",
        "Box: median (thick line) and quartiles; whiskers up to 1.5 x the interquartile range; all subjects drawn as dots.",
        "Read: medians rise from train to test (multisite 4.82 -> 6.28, wrist-finger 0.85 -> 1.18); hard subjects stay "
        "hard in every split. " + CAP,
    ],
}

PROTOCOL = ("Out-of-fold: each subject is tested once, by the 3 seeds of the fold whose test group it is in "
            "(never trained on it); gate limit and Kalman parameters tuned on that fold's val group.")


def card_line(head, text, style_p="P2", style_t="T50"):
    return f'<text:p text:style-name="{style_p}"><text:span text:style-name="{style_t}">{esc(head + text)}</text:span></text:p>'


summary_lines = [
    card_line("", "dist + boot · 3-seed ensemble · input gate · Kalman + RTS: MAE 3.64 BPM", "P37", "T49"),
    card_line("", "Subject-macro MAE, mean of the 2 sub-datasets, all 47 subjects, 428k labelled 10 s windows (every 2 s)"),
    card_line("", " "),
    card_line("Raw ensemble:  ", "multisite 6.74 · wrist-finger 1.48 · both 4.11 BPM (RMSE 7.94)"),
    card_line("Shipped pipeline, every window:  ", "multisite 5.82 · wrist-finger 1.47 · both 3.64 BPM (RMSE 6.54)"),
    card_line("Shipped, gated gaps ≤ 30 s only (98.5 %):  ", "multisite 5.66 · wrist-finger 1.47 · both 3.56 BPM"),
    card_line("", " "),
    card_line("Within 5 BPM:  ", "multisite 63 % · wrist-finger 96 % of windows; within 10 BPM: 77 % · 99.5 %"),
]
s1 = page("Results", 28,
          title("Results: winner model, out-of-fold test")
          + '<draw:rect draw:style-name="gr4" draw:text-style-name="P7" draw:layer="layout" svg:width="30.867cm" '
            'svg:height="10cm" svg:x="1.5cm" svg:y="4cm" draw:corner-radius="0.35cm"><text:p/></draw:rect>'
          + '<draw:frame draw:style-name="gr1" draw:text-style-name="P3" draw:layer="layout" svg:width="29.867cm" '
            f'svg:height="9.4cm" svg:x="2cm" svg:y="4.3cm"><draw:text-box>{"".join(summary_lines)}</draw:text-box></draw:frame>'
          + '<draw:frame draw:style-name="gr2" draw:text-style-name="P3" draw:layer="layout" svg:width="30.867cm" '
            'svg:height="2.4cm" svg:x="1.5cm" svg:y="14.5cm"><draw:text-box><text:p text:style-name="P6">'
            '<text:span text:style-name="T51">Protocol: </text:span>'
            f'<text:span text:style-name="T39">{esc(PROTOCOL)} The submitted model has no test data left; these numbers are its estimate.</text:span>'
            '</text:p></draw:text-box></draw:frame>',
          "Source: notebooks/final_results.ipynb (cache/results/final_dist_bootHL.csv). 'Shipped' = what estimate_heartrate returns: "
          "no output gate, gated windows bridged by the Kalman smoother.")

s2 = page("ResultsBA", 29,
          title("Bland-Altman: unbiased, wide limits on multisite")
          + image("bland_altman", 1.5, 4 + (12.6 - 21.5 * 924 / 2000) / 2, 21.5)
          + bullets([("Wrist-finger:", "bias +0.12 BPM, 95 % limits −4.3 / +4.5"),
                     ("Multisite:", "bias +1.2 BPM, limits −19.9 / +22.3"),
                     ("Diagonal bands:", "prediction stuck near 80 BPM while the true HR moves (the pull to the middle)"),
                     ("Data:", "all labelled test windows, shipped pipeline, density on log scale")])
          + takeaway("Controlled finger/wrist clips are near ECG quality; free-living multisite errors are large but not systematic on average."),
          "Windows overlap (10 s every 2 s): limits describe the error spread, not a sample-size-based CI.")

s3 = page("ResultsScatter", 30,
          title("Predicted vs ECG: the slope tells the story")
          + image("scatter", 1.5, 4 + (12.6 - 21.5 * 1134 / 2000) / 2, 21.5)
          + bullets([("Wrist-finger:", "r = 0.98, slope 0.93"),
                     ("Multisite:", "r = 0.72, slope 0.49: a horizontal band at 80-90 BPM"),
                     ("Raw vs shipped:", "multisite r 0.57 vs 0.72 (gate + Kalman)"),
                     ("Grey line:", "perfect agreement; red dashed: least-squares fit")])
          + takeaway("Where the PPG has no clear pulse the model falls back to the typical heart rate."),
          "Same windows as the Bland-Altman slide.")

s4 = page("ResultsBins", 31,
          title("Error per heart-rate bin: the tails stay hard")
          + image("bins", 1.5 + (21.5 - 12.6 * 2000 / 1344) / 2, 4, 12.6 * 2000 / 1344)
          + bullets([("80-90 BPM:", "MAE 2.2, bias −0.3"),
                     ("40-60 BPM:", "MAE 11.8, over-predicted by 11.4"),
                     ("130+ BPM:", "MAE 42.5, under-predicted by 42.4 (2,820 windows)"),
                     ("Smoothing:", "lowers MAE in every bin by about 1 BPM, cannot fix the bias")])
          + takeaway("The bootstrap and the dist head narrowed the pull to the middle; they did not remove it."),
          "Window-pooled MAE per ECG truth bin. Numbers above the bars: windows per bin.")

w = 15.2
h1, h2 = w * 756 / 2000, w * 714 / 2000
gap = (12.6 - h1 - h2) / 3
s5 = page("ResultsSubjects", 32,
          title("Per subject: wrist-finger uniform, multisite spread")
          + image("subjects", 1.5 + (21.5 - w) / 2, 4 + gap, w)
          + image("error_cdf", 1.5 + (21.5 - w) / 2, 4 + 2 * gap + h1, w)
          + bullets([("Median subject MAE:", "wrist-finger 1.14, multisite 5.30 BPM"),
                     ("Worst subject:", "multisite P14, 12.4 BPM"),
                     ("Every wrist-finger subject:", "≤ 3.4 BPM"),
                     ("Within 5 BPM:", "96 % (wrist-finger), 63 % (multisite) of windows")])
          + takeaway("Multisite error is spread over many subjects, not one outlier."),
          "Top: per-subject MAE sorted. Bottom: cumulative share of windows by absolute error, raw vs shipped.")

TT_NOTE = ("Raw 3-seed ensemble per fold, no gate or smoothing; real capped windows of cache/windows/ "
           "(no bootstrap, no noise). Train = the windows training sampled from.")
s6 = page("TrainTestFolds", 0,
          title("Train vs test: 5 fold ensembles")
          + image("train_test_folds", 1.5, 4 + (12.6 - 21.5 * 798 / 2000) / 2, 21.5)
          + bullets([("Both datasets:", "train 3.39 ± 0.27, val 4.00 ± 0.37, test 4.18 ± 0.27 BPM (mean ± sd over folds)"),
                     ("Multisite:", "train 5.71 vs test 6.88"),
                     ("Wrist-finger:", "train 1.08 vs test 1.48"),
                     ("Gap:", "about 0.8 BPM; even on train, multisite stays near 6 BPM")])
          + takeaway("Mild over-fitting; multisite error is mostly PPG quality, not memorisation."),
          TT_NOTE + " Subject-macro MAE per sub-dataset. Source: notebooks/final_results.ipynb.")
s7 = page("TrainTestSubjects", 0,
          title("Train vs test per subject")
          + image("train_test_subjects", 1.5, 4 + (12.6 - 21.5 * 840 / 2000) / 2, 21.5)
          + bullets([("Median subject MAE:", "multisite 4.82 / 6.03 / 6.28, wrist-finger 0.85 / 1.03 / 1.18 (train / val / test)"),
                     ("Lines:", "the same subject across splits"),
                     ("Hard subjects", "stay hard in every split")])
          + takeaway("The ranking of subjects barely changes between train and test."),
          TT_NOTE + " Train value = mean over the 3 folds in which the subject was in train.")

# Context notes for the slides before the results (appended after any existing note text; background, not a script)
EXTRA = {
    "Slide1": [
        "Status 2026-11-09: results are in (slides 28-34). Submitted model: ensemble of all 15 CV models "
        "(5 folds x 3 seeds of dist + boot); its performance estimate is the out-of-fold CV result.",
    ],
    "Slide2": [
        "Part 6 now holds real numbers: out-of-fold test results of the winner (28-32) and train vs test of the 5 fold "
        "ensembles (33-34).",
    ],
    "Slide3": [
        "Why regression per window: the label is the mean ECG heart rate over the same 10 s, so each window is one "
        "independent example; temporal context across windows only enters later through the smoother.",
        "Why subject-wise: windows of one subject are strongly correlated (same physiology, same sensor fit). A window-"
        "wise split would leak the subject into test and overstate accuracy.",
        "Exercise set (wrist-ppg-during-exercise): 19 files, 8 subjects, the only set with accelerometer data; dropped "
        "because only 38 % of its windows had usable ECG labels (README.md).",
    ],
    "Slide6": [
        "Window counts after QC: 51,864 cached windows (multisite capped). The same curves (cache/hr_curves/) also "
        "label the 2 s sequences used for smoothing.",
    ],
    "Slide5": [
        "Order matters: the band-pass runs on the whole recording before windowing, so window edges carry no filter "
        "transients; z-scoring happens per window at load time, which removes amplitude differences between sensor sites.",
        "Band 0.5-8 Hz: keeps the pulse fundamental (0.5 Hz = 30 BPM and up) and the harmonics that shape the wave; "
        "baseline wander (breathing, posture) and high-frequency noise are cut.",
        "Code: src/bpm/preprocessing.py (bandpass_filter), windowing.py (make_windows), cache.py (per-subject npz, cap 2000 "
        "random windows, seed 42), dataset.py (zscore_window).",
    ],
    "Slide9": [
        "Why balancing: without it multisite (about 74 % of train windows) would dominate every epoch and wrist-finger "
        "would be under-fitted. Weight 1 / n per sub-dataset gives each set half of the samples.",
        "The noise level 0.05 is small relative to the z-scored signal (SD 1): it regularises without hiding the pulse.",
        "Code: src/bpm/dataset.py (sampler_weights, _augment).",
    ],
    "Slide10": [
        "Why it is needed: only 2.8 % of windows are at 110 BPM or more, so a loss that averages over windows barely sees "
        "them. The bootstrap adds plausible examples there instead of re-weighting the few real ones.",
        "Effect in the CV (Stage B/C): with the regression head it helped the tails but hurt the middle; with the dist "
        "head it was part of the overall winner.",
    ],
    "Slide12": [
        "Design choice: a small CNN for local pulse shape plus a recurrent layer for beat-to-beat rhythm is a common "
        "pattern for PPG heart-rate models; the size was kept small (about 160k parameters) because there are only 47 "
        "subjects.",
        "Mean pooling over time fits the label definition (mean HR over the window) and makes the output independent of "
        "where in the window a beat falls.",
        "Code: src/bpm/model.py (CNNBiLSTM).",
    ],
    "Slide13": [
        "Same table applies to every trained model; checkpoints are about 0.6 MB each (15 checkpoints in the submitted "
        "ensemble).",
    ],
    "Slide22": [
        "Ensemble result in the CV (test, mean over folds): dist + boot single model 4.37 BPM, 3-seed ensemble 4.18. "
        "The gain (about 0.2 BPM) was similar for every config.",
    ],
    "Slide23": [
        "Result in the CV: the dist curve std ranked errors best (Spearman rho 0.71 on test, AURC 1.86). At 90 % val "
        "coverage it kept 89 % of test windows and lowered their MAE from 4.18 to 3.40 BPM (dist + boot ensemble).",
        "The submitted estimate_heartrate cannot abstain (the interface needs a value every 0.5 s), so the gate is used "
        "only as the input gate of the smoother: unsure windows get no weight and are bridged.",
    ],
    "Slide24": [
        "Result in the CV (dist + boot, test, mean over folds): raw ensemble 4.11 BPM on all windows, + Kalman 3.69, "
        "+ input and output gate 3.06 at 86 % coverage. Kalman and HMM were close; Kalman was chosen.",
        "Kalman parameters tuned per fold on val: m (process-noise multiplier) 1-4, c (measurement-noise scale) 0.5-4.",
    ],
    "Slide17": [
        "One training run takes about 2-3 minutes on the RTX 3080 (about 2.5 s per epoch). Best epochs of the 15 "
        "dist + boot runs: 6-37, median 18.",
        "Code: scripts/train.py; each run writes a sidecar cache/runs/<tag>.json with config, best/stop epoch, best val "
        "and git hash.",
    ],
    "Story1": [
        "This part summarises docs/cv_story.odp (built from notebooks/cv_selection.ipynb and cv_gate_smoothing.ipynb). "
        "Selection only ever used validation errors; test folds were looked at only to report.",
        "LTGO (leave-test-group-out): for test group k, the decision uses val errors of the other folds only, so group k's "
        "own data never influences the choice that is scored on it. The 5 LTGO decisions are a stability check of the "
        "global decision.",
    ],
    "Story2": [
        "Counts from cache/windows/ (after ECG QC, multisite capped at 2000 windows per subject); see the data statistics "
        "slide for the per-dataset breakdown.",
        "The crowded middle is the root cause of the pull toward 75-85 BPM shown in the next slides: errors there dominate "
        "any average loss.",
    ],
    "Slide21": [
        "Statistics behind every decision: per-subject val MAE, median over the 3 seeds, paired against the best config; "
        "key = mean over the 2 sub-datasets; standard error (SE) by bootstrap over subjects (stratified by sub-dataset). "
        "Tie set = configs within 1 SE of the best.",
        "Test comparisons vs the MSE baseline: Wilcoxon signed-rank on paired per-subject out-of-fold MAE per "
        "sub-dataset, Holm-adjusted across configs.",
    ],
    "Story3": [
        "Key = paired difference in per-subject val MAE vs the best config (here MSE), mean over sub-datasets; SE from a "
        "subject bootstrap. A difference inside 1 SE counts as a tie.",
        "Out-of-fold test MAE (subject-macro, mean over seeds): MSE 4.64, Huber delta 10 4.72, MAE 4.87 BPM.",
    ],
    "Story4": [
        "Slope = least-squares fit of prediction on truth over all test windows of the MSE runs; 1 would be perfect, "
        "0 a constant prediction.",
        "Classic regression to the mean under label imbalance: when the PPG is ambiguous, predicting the common value "
        "minimises the average loss.",
    ],
    "Story5": [
        "Out-of-fold test MAE: MSE + boot 4.98 vs MSE 4.64 BPM (Holm-adjusted Wilcoxon: multisite p = 0.014, "
        "wrist-finger p = 0.021 against MSE).",
        "Interpretation: a single regression output must trade off: shifting predictions toward the tails moves the "
        "middle too.",
    ],
    "Story6": [
        "Out-of-fold test MAE: dist + boot 4.37 vs MSE 4.64 BPM. Per sub-dataset vs MSE (Holm-adjusted Wilcoxon): "
        "wrist-finger p = 0.0008, multisite p = 0.31, so the gain is clear on wrist-finger and not significant on "
        "multisite.",
        "Bin-balanced MAE = mean of the per-bin MAEs (40-60, 60-70, ..., 130+), so every heart-rate range counts "
        "equally; it measures the tails, the subject-macro MAE measures the typical case.",
        "Why it helps: the soft-target cross-entropy does not average two candidate rates into a middle value; the "
        "decode takes the main peak.",
    ],
    "Winner1": [
        "Submitted version (estimate_heartrate): the same per-window model, but all 15 CV checkpoints (5 folds x 3 "
        "seeds) are averaged, and there is no output gate because the interface must return a value every 0.5 s; "
        "gated windows are bridged by the Kalman smoother.",
    ],
    "Winner2": [
        "Not done: a refit on all 47 subjects. The 15 fold models together saw every subject, and the CV result is the "
        "performance estimate of this recipe; a refit would need a fixed epoch count (median best epoch 18) and a "
        "train.py change.",
    ],
}


def add_context(page_xml, paras):
    head, sep, tail = page_xml.partition("<presentation:notes")
    assert sep, "page without notes"
    add = "".join(f"<text:p>{esc(t)}</text:p>" for t in paras)
    if "<draw:text-box/>" in tail:
        tail = tail.replace("<draw:text-box/>", f"<draw:text-box>{add}</draw:text-box>", 1)
    else:
        i = tail.rindex("</draw:text-box>")
        tail = tail[:i] + add + tail[i:]
    return head + sep + tail


pages = re.findall(r'<draw:page [^>]*>.*?</draw:page>', content, re.S)
for p in pages:
    name = re.search(r'draw:name="([^"]*)"', p).group(1)
    if name in EXTRA:
        content = content.replace(p, add_context(p, EXTRA.pop(name)))
assert not EXTRA, f"slides not found: {list(EXTRA)}"
pages = re.findall(r'<draw:page [^>]*>.*?</draw:page>', content, re.S)
old = [p for p in pages if 'draw:name="Results"' in p]
assert len(old) == 1, len(old)
content = content.replace(old[0], s1 + s2 + s3 + s4 + s5)
n_pages = len(pages) + 4
s6 = s6.replace('draw:page-number="0"', f'draw:page-number="{n_pages + 1}"')
s7 = s7.replace('draw:page-number="0"', f'draw:page-number="{n_pages + 2}"')
last = re.findall(r'<draw:page [^>]*>.*?</draw:page>', content, re.S)[-1]
content = content.replace(last, last + s6 + s7)
assert content.count("Final results (to follow)") == 1
content = content.replace("Final results (to follow)", "Final results (out-of-fold CV)")
(X / "content.xml").write_text(content)

for name in IMG_W:
    shutil.copy(PLOTS / f"{name}.png", X / "Pictures" / f"final_{name}.png")
    entry = f'<manifest:file-entry manifest:full-path="Pictures/final_{name}.png" manifest:media-type="image/png"/>'
    if entry not in manifest:
        manifest = manifest.replace("</manifest:manifest>", entry + "</manifest:manifest>")
(X / "META-INF" / "manifest.xml").write_text(manifest)
print("ok")
