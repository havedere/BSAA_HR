// helpers for the block diagrams
#let blk(title, body, fill: luma(235)) = box(
  fill: fill, inset: 6pt, radius: 3pt, width: 100%,
  align(center)[#text(9pt, weight: "bold")[#title] \ #text(8pt)[#body]],
)
#let arr = align(horizon + center, text(12pt)[→])
#let down = align(center, text(12pt)[↓])

= Introduction

Photoplethysmography (PPG) measures the change of blood volume in the tissue optically. A light source illuminates the skin and a photodiode records the transmitted or reflected light, which is modulated by every heartbeat @allen2007. Because the sensor is cheap and small, PPG is built into watches, rings and earbuds. Its main weakness is noise: motion, changing contact pressure and low perfusion distort or hide the pulse wave, and the reference measurement for heart rate (HR) remains the electrocardiogram (ECG).

Classic PPG heart-rate estimation finds the dominant frequency in a short window or detects individual pulses, often combined with accelerometer-based motion cancellation such as TROIKA @zhang2015troika. Deep learning has replaced these hand-made pipelines in recent work. Convolutional neural networks (CNN) learn the shape of the pulse wave directly from the signal or its spectrum (DeepPPG @reiss2019deepppg), and CNN-LSTM hybrids add a recurrent layer that models the rhythm across several beats (CorNET @biswas2019cornet). A long short-term memory (LSTM) network @hochreiter1997lstm keeps a state over the sequence. A bidirectional LSTM (BiLSTM) @schuster1997brnn reads the sequence forwards and backwards, so each time step sees the beats before and after it. BeliefPPG @bieri2023beliefppg predicts a probability distribution over HR per window and smooths it over time, which also yields an uncertainty estimate.

This report describes a CNN-BiLSTM that estimates HR from a single PPG channel, without accelerometer data. The model predicts a distribution over HR. An ensemble, a quality gate and a Kalman smoother turn the per-window distributions into a continuous HR trace at 2 Hz.

= Data

Two sub-datasets with simultaneous ECG and PPG are used, both sampled at 100 Hz and stored as EDF files without annotations:

- *Multisite-PPG*: 20 subjects, 519 free-living recordings from four wearables (earring, necklace, ring, watch). Recording length median 4.5 h, partly over 23 h.
- *Wrist-finger-PPG*: 27 subjects with 6 short controlled recordings each (165 files), median 4.0 min (1.8–7.0 min).

A third sub-dataset (wrist PPG during exercise, 8 subjects) was excluded, because only 38 % of its windows had a usable ECG label after quality control (@sec:label).

After labelling and windowing (@sec:input), 51 864 windows remain: 37 668 multisite and 14 196 wrist-finger. The label distribution is centred at 60–90 BPM (72 % of all windows); only 2.8 % of the windows are at 110 BPM or more, almost all of them from multisite. The mean label is 83.3 ± 14.6 BPM (multisite) and 79.2 ± 10.4 BPM (wrist-finger).

= Methods

@fig:pipeline gives an overview. The ECG is used only to produce the label; the network sees only the PPG.

#figure(
  grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 3pt, row-gutter: 6pt,
    blk[ECG][R-peaks, RR checks, quality gate → 1 Hz HR curve], arr,
    blk[Label][mean HR over the 10 s window], arr,
    blk[Training][CE loss on Gaussian target, 5 folds × 3 seeds], arr,
    blk(fill: luma(215))[15 models][CNN-BiLSTM, dist head],
    blk[PPG][band-pass 0.5–8 Hz, zero-phase], arr,
    blk[Windows][10 s, 2 s stride, z-score], arr,
    blk[Ensemble][mean softmax curve → BPM + curve std], arr,
    blk(fill: luma(215))[Post-processing][input gate → Kalman + RTS → 2 Hz],
  ),
  caption: [Overview. Top row: label path (training only). Bottom row: input and inference path.],
) <fig:pipeline>

== From ECG to label <sec:label>

*R-peak detection.* R-peaks are detected in the raw ECG with the Pan-Tompkins algorithm @pan1985 as implemented in NeuroKit2 @makowski2021neurokit2. For each pair of consecutive peaks, the instantaneous HR is $60 \/ "RR"$ (RR in seconds), assigned to the time of the second peak. These values are linearly interpolated onto a 1 Hz grid; seconds before the first or after the last anchor are undefined.

*Beat checks.* A beat is rejected if its instantaneous HR lies outside 30–220 BPM, or if its RR interval differs from the previous one by more than a factor of 1.2 (ratio $> 1.2$ or $< 1 \/ 1.2$). All integer seconds between the two peaks of a rejected beat (inclusive) are set to undefined. No value is interpolated across a rejected beat.

*ECG quality gate.* Pan-Tompkins produces a regular train of false beats on lead-off ramps that passes the beat checks. Two further rules catch this:
- _Beat agreement (bSQI)_ @li2008bsqi: a second detector (NeuroKit2 default method on the NeuroKit2-cleaned ECG) runs on the same recording. In each non-overlapping 10 s block, beats of the two detectors are matched within 150 ms. If $"matched" \/ (n_1 + n_2 - "matched") < 0.8$, or either detector finds fewer than two beats, the whole block is undefined.
- _Flat ECG_: a second is undefined if the peak-to-peak amplitude of the cleaned ECG in it is at most 10 % of the recording median.

Finally, every second in which the raw PPG is constant (sensor off, padding) is undefined, even if the ECG is valid.

*Label.* The label of a window is the mean of the ten 1 Hz HR values it covers. A window containing any undefined second is discarded.

== From PPG to model input <sec:input>

*Filtering.* The full PPG recording is band-pass filtered with a 4th-order Butterworth filter, 0.5–8 Hz, applied forwards and backwards (zero phase). The band covers HR from 30 BPM up to the higher harmonics of the pulse wave.

*Windowing.* Windows of 10 s (1000 samples) start every 2 s. Only windows with a valid label (@sec:label) are kept for training.

*Per-subject cap.* Multisite recordings are long, so a few subjects would dominate the training data. For each multisite subject, at most 2000 windows are kept, drawn uniformly at random without replacement (seed 42), first per file and then per subject. Wrist-finger subjects are not capped (median 541 windows).

*Normalisation.* Each window is z-scored individually (mean 0, standard deviation 1). Windows with zero standard deviation are left unchanged. This removes the amplitude, which depends on sensor, site and contact, and keeps only the shape.

== Augmentation and balancing (training only)

*Gaussian noise.* Every time a training window is drawn, noise with $sigma = 0.05$ (in z-scored units) is added. Validation and test windows are never augmented.

*Balanced sampling.* The training set holds about three times more multisite than wrist-finger windows. Batches are drawn with a weighted random sampler with replacement, in which each sub-dataset has the same total weight, so each fills about half of every epoch. One epoch draws as many windows as there are real training windows.

*Time-warp bootstrap.* Low and high HR are rare (@fig:bins), and a model trained on the raw distribution pulls its predictions towards the middle. Synthetic windows are therefore created by time-warping real PPG @um2017augmentation (@fig:warp). A source span of $10 dot f$ seconds is linearly resampled into one 10 s window. For $f > 1$ the pulse is compressed and the HR rises, for $f < 1$ it falls. The label is $f$ times the HR curve sampled at the ten source times $t_0 + f k$, $k = 0, dots, 9$, averaged. The span is extended by $3 f$ seconds on each side, filtered again with the same band-pass after warping, and cropped. The second filter removes energy that warping moves out of the 0.5–8 Hz band and that would otherwise mark the window as synthetic.

#figure(
  grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 3pt,
    blk[Source span][$10 dot f$ s of real PPG, $f in [0.7, 1.6]$], arr,
    blk[Resample][squeeze or stretch into 10 s], arr,
    blk[Re-filter][pad 3 s per side, band-pass, crop], arr,
    blk[Label][$f times$ source HR],
  ),
  caption: [Time-warp bootstrap for synthetic low- and high-HR windows.],
) <fig:warp>

Synthetic windows are generated from the training subjects of each fold only. Target bins are 40–50 and 50–60 BPM (both sub-datasets) and 10 BPM bins from 110 to 190 BPM (multisite) or 110 to 130 BPM (wrist-finger). Each bin is filled up to 500 windows per sub-dataset, counting the real training windows already in it. Candidate source starts are integer seconds with a valid 10 s mean HR $h$. For a bin $[l, u)$, the factor is drawn uniformly from $[max(l \/ h, 0.7), min(u \/ h, 1.6)]$, and the window is kept only if its label falls into the bin and no second of the source span is undefined. Each source start is used at most five times; a bin that runs out of sources stays short. For the high bins, a source is used only if the dominant frequency of its PPG (Hann-windowed FFT, 0.6–3.5 Hz) lies within 10 % of its ECG HR. Otherwise motion-corrupted PPG would receive a high label without a matching pulse. In the sampler, synthetic windows receive at most 20 % of their sub-dataset's weight.

== Model architecture

The network maps one z-scored 10 s window ($1 times 1000$) to 191 logits, one per HR bin of 1 BPM from 30 to 220 BPM. @tab:arch lists the layers; the model has 161 791 parameters.

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, left, right),
    stroke: none,
    table.hline(),
    table.header([*Block*], [*Operation*], [*Output*], [*Params*]),
    table.hline(stroke: 0.5pt),
    [Input], [10 s band-passed PPG at 100 Hz, z-scored], [$1 times 1000$], [–],
    [Conv 1], [Conv1d 1→32, kernel 5, same padding · BatchNorm · ReLU · MaxPool 2], [$32 times 500$], [256],
    [Conv 2], [Conv1d 32→64, kernel 5 · BatchNorm · ReLU · MaxPool 2], [$64 times 250$], [10 432],
    [Conv 3], [Conv1d 64→128, kernel 5 · BatchNorm · ReLU · MaxPool 2], [$128 times 125$], [41 344],
    [BiLSTM], [1 layer, 64 hidden units per direction], [$125 times 128$], [99 328],
    [Pooling], [mean over the 125 time steps], [128], [–],
    [Dense], [Linear 128→32 · ReLU · Dropout 0.3], [32], [4 128],
    [Head], [Linear 32→191 (logits)], [191], [6 303],
    table.hline(),
  ),
  caption: [CNN-BiLSTM layers.],
) <tab:arch>

*CNN encoder.* Three convolution blocks with batch normalisation @ioffe2015bn extract the local pulse shape (upstroke, peak, dicrotic notch) and suppress noise. Each max-pooling halves the time axis, from 1000 to 125 steps (80 ms apart); one step sees 36 input samples (0.36 s).

*BiLSTM.* The sequence of 125 feature vectors is read by a bidirectional LSTM @hochreiter1997lstm @schuster1997brnn. It models the rhythm across beats, so a corrupted beat can be inferred from its neighbours.

*Pooling and head.* The BiLSTM outputs are averaged over time, which matches the label (mean HR over the window). A dense layer with dropout @srivastava2014dropout and a linear layer produce the 191 logits.

*Distribution head.* The task is treated as label distribution learning @gao2017ldl. The softmax of the logits gives a probability curve $p(c)$ over the bin centres $c = 30, dots, 220$. The training target for a label $y$ is a discretised Gaussian
$ q(c) = exp(-(c - y)^2 \/ (2 sigma^2)) \/ sum_(c') exp(-(c' - y)^2 \/ (2 sigma^2)), quad sigma = 2 "BPM". $
Neighbouring bins share the credit, so a prediction 1 BPM off is penalised less than one 30 BPM off. To decode one value, the bins within ±10 BPM of the curve's maximum are kept and their expected value is taken:
$ hat(y) = (sum_(|c - c^*| <= 10) p(c) c) \/ (sum_(|c - c^*| <= 10) p(c)), quad c^* = arg max_c p(c). $
A plain mean over all bins would be pulled towards a second mode (harmonic or motion peak), and the argmax alone is quantised to 1 BPM. The standard deviation of the whole curve, $s = sqrt(sum_c p(c) (c - mu)^2)$ with $mu = sum_c p(c) c$, serves as the uncertainty score of the window.

== Training

*Loss.* Cross-entropy between target and prediction per window, $cal(L) = -sum_c q(c) log p(c)$, averaged over the batch.

*Optimisation.* Adam @kingma2015adam, learning rate $10^(-3)$, batch size 64, at most 100 epochs. The learning rate is halved when the validation metric has not improved for 5 epochs (minimum $10^(-5)$).

*Validation and early stopping.* After each epoch, the model predicts all validation windows (capped, not augmented). The validation metric is the balanced MAE: the window-pooled MAE of each sub-dataset, averaged over the two sub-datasets. Training stops after 20 epochs without improvement; the checkpoint with the lowest balanced validation MAE is kept, not the last one.

*Seeds.* Each fold is trained with three seeds (42, 1, 2), which set the weight initialisation, the sampling order and the noise. cuDNN runs in deterministic mode. Training ran on one NVIDIA RTX 3080 GPU.

== Cross-validation <sec:cv>

All splits are subject-wise: no subject's windows appear in more than one of train, validation and test. Within each sub-dataset, subjects are assigned to five groups, stratified by the share of their windows at high HR (≥ 130 BPM for multisite, ≥ 100 BPM for wrist-finger). Subjects are sorted by this share, cut into strata of five, shuffled within each stratum (seed 42) and dealt to groups 0–4. This puts the subjects with the most high-HR windows into different groups. Multisite has 4 subjects per group, wrist-finger 6/6/5/5/5.

In fold $k$, group $k$ is the test set, group $(k + 1) mod 5$ the validation set and the remaining three groups the training set. Every subject is thus tested once and validated once. Everything that is fitted (bootstrap quotas, gate limit, smoother parameters) uses only the training or validation subjects of the fold. Five folds times three seeds give 15 trained models.

== Inference and post-processing

#figure(
  grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 3pt,
    blk[Ensemble][mean of the softmax curves, decode once], arr,
    blk[BPM + score][$hat(y)_t$ and curve std $s_t$ per window], arr,
    blk[Input gate][$s_t > 16.205$ → no weight], arr,
    blk[Kalman + RTS][forward filter, backward smoother → 2 Hz],
  ),
  caption: [Post-processing of the window predictions of one recording.],
) <fig:post>

*Input.* A raw PPG signal of any sampling rate is resampled to 100 Hz (polyphase), missing samples are linearly interpolated, and the signal is band-pass filtered and cut into 10 s windows every 2 s as in @sec:input.

*Ensemble.* All 15 models predict every window; their softmax curves are averaged @lakshminarayanan2017ensembles, and the mean curve is decoded once into $hat(y)_t$ and its standard deviation $s_t$.

*Input gate.* A window with $s_t > 16.205$ BPM is treated as unreliable. The limit comes from cross-validation: for each fold, the limit is the 90 % quantile of $s_t$ over that fold's validation windows (3-seed ensemble); the shipped limit is the median of the five fold limits.

*Kalman filter and RTS smoother.* The HR over the window sequence of one recording is modelled as a random walk @kalman1960:
$ x_t = x_(t-1) + w_t, quad w_t tilde cal(N)(0, Q), quad Q = (m dot sigma_"step")^2, $
$ hat(y)_t = x_t + v_t, quad v_t tilde cal(N)(0, R_t), quad R_t = (c dot s_t)^2. $
The step deviation $sigma_"step" = 1.0$ BPM per 2 s step is estimated robustly from the training labels (10 s mean labels at 2 s stride, $1.4826 dot "MAD"$ of the first differences). Gated windows get $R_t = infinity$, so the filter bridges them with the prediction alone. The filter starts at the first ungated measurement with variance $10^6$. A forward Kalman pass is followed by a backward Rauch-Tung-Striebel (RTS) pass @rauch1965rts, so each estimate uses past and future windows. The factors $m in {1, 2, 4}$ and $c in {0.5, 1, 2, 4}$ were tuned on the validation subjects (lowest subject-macro balanced MAE); the shipped values are $m = 1$, $c = 0.5$.

*Output.* The smoothed values sit at the window centres (start + 5 s). They are linearly interpolated to a 2 Hz output grid; before the first and after the last centre the nearest value is held. No output gate is applied, because the interface requires a value every 0.5 s.

== Evaluation

The shipped 15-model ensemble has seen every subject during training and cannot be tested fairly. The performance of the recipe is therefore estimated out-of-fold: each subject is predicted by the 3-seed ensemble of the fold in which it is a test subject, with the gate limit and smoother parameters of that fold. The pipeline runs on every 2 s window within ±5 min of a labelled window, without the per-subject cap, and is scored on all 427 994 labelled windows.

The primary metric is the subject-macro MAE: the MAE per subject, averaged over the subjects of a sub-dataset, then averaged over the two sub-datasets. A long recording thus counts no more than a short one. Further metrics are the window-pooled MAE, the RMSE, the bias (mean of prediction minus truth), Pearson's $r$, the share of windows within ±5 and ±10 BPM, and Bland-Altman limits of agreement @bland1986.

= Results

@tab:results compares the raw ensemble with the shipped pipeline. Over both sub-datasets, the subject-macro MAE drops from 4.11 to 3.64 BPM through gate and smoother. On wrist-finger the model is almost exact (1.47 BPM, $r = 0.98$); on multisite it reaches 5.82 BPM ($r = 0.72$). Gating and smoothing help mainly on multisite (6.73 → 5.82 BPM, $r$ 0.57 → 0.72).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*Pipeline*], [*Dataset*], [*MAE#sub[subj]*], [*MAE#sub[win]*], [*RMSE*], [*Bias*], [*r*], [*±5 / ±10*]),
    table.hline(stroke: 0.5pt),
    [Raw ensemble], [Multisite], [6.73], [7.49], [13.35], [+1.81], [0.57], [62 / 74 %],
    [], [Wrist-finger], [1.48], [1.50], [2.53], [+0.16], [0.97], [95 / 99 %],
    [], [Both], [4.11], [4.49], [7.94], [+0.99], [0.77], [78 / 87 %],
    table.hline(stroke: 0.5pt),
    [Shipped], [Multisite], [5.82], [6.47], [10.84], [+1.22], [0.72], [63 / 77 %],
    [], [Wrist-finger], [1.47], [1.48], [2.24], [+0.12], [0.98], [96 / 99.5 %],
    [], [Both], [*3.64*], [3.97], [6.54], [+0.67], [0.85], [80 / 89 %],
    table.hline(),
  ),
  caption: [Out-of-fold test results (BPM) on 427 994 labelled windows of 47 subjects. Shipped = ensemble + input gate + Kalman/RTS. MAE#sub[subj]: subject-macro MAE; "Both" averages the two sub-datasets for MAE#sub[subj], other columns pool all windows.],
) <tab:results>

The input gate removes 1.5 % of the multisite windows and none of the wrist-finger windows. Scoring only windows whose gated gaps are at most 30 s (98.5 % of the windows) gives 3.56 BPM.

@fig:scatter shows predictions against the ECG label. On wrist-finger the points follow the diagonal (slope 0.93). On multisite the fit has a slope of 0.49: a large share of windows is predicted near 80–90 BPM regardless of the true HR. The Bland-Altman analysis (@fig:ba) confirms that the bias is small (+1.2 BPM multisite, +0.1 BPM wrist-finger), but the multisite limits of agreement are wide (−19.9 to +22.3 BPM, wrist-finger −4.3 to +4.5 BPM).

#figure(
  image("../figures/scatter.png", width: 100%),
  caption: [Predicted HR against ECG label, shipped pipeline, all test windows (log density). Grey: identity; red dashed: least-squares fit.],
) <fig:scatter>

#figure(
  image("../figures/bland_altman.png", width: 100%),
  caption: [Bland-Altman plots, shipped pipeline. Solid: bias; dashed: 95 % limits of agreement.],
) <fig:ba>

@fig:bins breaks the error down by true HR. Between 70 and 90 BPM, where most data lies, the MAE is 2.2–3.7 BPM. Towards the tails the error grows and becomes a bias: 40–60 BPM is over-predicted by 11.4 BPM, and HR above 130 BPM is under-predicted by 42.4 BPM. Smoothing lowers the MAE in every bin by about 1 BPM but does not remove this bias.

#figure(
  image("../figures/bins.png", width: 85%),
  caption: [Error per HR bin, test windows. Top: MAE of raw ensemble and shipped pipeline (number of windows above the bars). Bottom: bias of the shipped pipeline.],
) <fig:bins>

Per subject (@fig:subjects), every wrist-finger subject stays below 3.4 BPM (median 1.14 BPM). Multisite subjects spread from about 2 to 12.4 BPM (median 5.30 BPM); the error is spread over many subjects rather than caused by one outlier. Across the five fold ensembles, the subject-macro MAE is 3.39 ± 0.27 BPM on training, 4.00 ± 0.37 BPM on validation and 4.18 ± 0.27 BPM on test subjects (raw ensembles, mean ± sd over folds).

#figure(
  image("../figures/subjects.png", width: 100%),
  caption: [Per-subject MAE, out-of-fold test, shipped pipeline, sorted.],
) <fig:subjects>

= Discussion and Conclusion

The CNN-BiLSTM estimates HR from controlled finger and wrist recordings with an error close to the ECG reference (1.47 BPM, 96 % of windows within 5 BPM). Free-living multisite PPG is much harder (5.82 BPM). Even on its own training subjects, the multisite error stays near 6 BPM (5.7 BPM on train vs. 6.9 BPM on test), so the gap is mainly due to signal quality, not overfitting. The authors of the multisite dataset report 2.3–8.7 BPM on the same recordings depending on the body site; their protocol differs, so the numbers are only roughly comparable.

The main failure mode is the pull towards the typical HR. When the PPG shows no clear pulse, the model falls back on the most common labels (80–90 BPM), which produces the horizontal band in @fig:scatter and the strong bias at low and high HR in @fig:bins. The time-warp bootstrap and the distribution head reduce this effect but do not remove it. Warping also scales systole and diastole equally, whereas a real HR increase mainly shortens diastole, so the synthetic high-HR windows are not fully realistic.

Gate and Kalman smoother lower the MAE by 0.5 BPM without retraining and give each estimate an uncertainty. Since the interface requires a value every 0.5 s, unreliable stretches are bridged by the smoother rather than flagged; a user of the output cannot tell these values apart.

Limitations: no accelerometer input, so motion cannot be modelled directly; labels depend on ECG R-peak detection, and the quality rules discard data rather than repairing it; only 47 subjects from two sources; the shipped ensemble is not refitted on all subjects, and its performance is estimated from the 3-seed fold ensembles. Future work should add motion signals, more high-HR data from real activity, and a time-domain output that can mark uncertain stretches.
