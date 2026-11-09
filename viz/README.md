# Model visualization

Tools to look inside the trained `CNNBiLSTM`: graph, weights, activations, math. Separate from the training pipeline; nothing in `src/` or `scripts/` depends on this folder. Outputs go to `cache/viz/` (gitignored).

## Setup

```bash
.venv/bin/pip install -r viz/requirements.txt   # onnx, onnxscript, treescope
.venv/bin/python viz/export_onnx.py --ckpt cache/best_huber10_fold0_seed42.pt [--subject multisite_P1]
```

`export_onnx.py` writes:
- `cache/viz/<ckpt stem>.onnx`: the model graph with weights, for Netron and Zetane.
- `cache/viz/<subject>_window.npy`: the subject's first cached window, z-scored, shape `(1, 1, 1000)` (10 s at 100 Hz). Use it as a real input in Zetane.

Any checkpoint works (`reg`, `dist`, `gauss`); the head is read from the weights, as in `scripts/evaluate.py`.

What ONNX changes compared to PyTorch:
- BatchNorm is folded into the preceding Conv, so the graph shows `Conv, Relu, MaxPool` x3. The unfused BatchNorm parameters are visible in treescope.
- The BiLSTM becomes a single `LSTM` op with `W`/`R`/`B` inputs. The `Slice`/`Concat` nodes in front of it reorder PyTorch's gate order `i,f,g,o` into ONNX's `i,o,f,c`.

## Netron: graph, shapes, weights

- Browser: open https://netron.app and drop in the `.onnx` file. The file is read locally in the browser, not uploaded.
- Desktop: AppImage from https://github.com/lutzroeder/netron/releases.

Click a node to see its attributes, input/output shapes and weight tensors. A raw `.pt` file only shows the state_dict, not the graph, so use the `.onnx`.

## Zetane Viewer: activations in 3D

Download from https://zetane.com (open-source viewer: https://github.com/zetane/viewer). Load the `.onnx` file, then load `<subject>_window.npy` as input to see each layer's feature maps for that window.

## treescope: module tree, weights, activations in Jupyter

```bash
.venv/bin/jupyter lab viz/treescope.ipynb
```

The notebook shows the module tree with every parameter, the activations of each layer for one real window (via forward hooks), the conv feature maps, and the LSTM input weights split by gate next to the gate equations.

treescope 0.1.10 reads `Tensor.names`, which torch 2.14 no longer has. The notebook's first cell adds a one-line shim for this.

## Other tools (not set up)

Loading your own model:
- Google Model Explorer (`pip install ai-edge-model-explorer`): op-level graph from `torch.export`, expandable layers, custom per-node data overlays (for example activation statistics). Handles large graphs better than Netron.
- TensorBoard: `SummaryWriter.add_graph` for the graph, plus forward hooks to log activation and weight histograms per epoch.

Explainers with step-by-step math (fixed example models, cannot load ours):
- CNN Explainer, https://poloclub.github.io/cnn-explainer: convolution, ReLU and pooling arithmetic per step.
- Transformer Explainer (poloclub) and LLM Visualization, https://bbycroft.net/llm: the same idea for transformers.
- LSTMVis (Harvard NLP): hidden-state dynamics of LSTMs.
- Colah, "Understanding LSTM Networks": the gate equations, explained.
- TensorFlow Playground, https://playground.tensorflow.org: small dense nets, trained live.

Why the model predicts what it does:
- Captum (PyTorch): Integrated Gradients and saliency on the input; shows which parts of the PPG waveform drive the BPM estimate.
- Lucent (PyTorch port of Lucid): feature visualization, built mostly for image CNNs, so less useful for 1D PPG.
