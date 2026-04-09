# MNIST CNN Project Plan

## Overview

Train a convolutional neural network to recognize handwritten digits (0–9) using the MNIST dataset and PyTorch. The trained model will be served via a Flask API, allowing a touch screen laptop to act as a drawing pad — sending drawn digits to the server and displaying predictions in real time.

---

## Goals

- Keep the codebase **modular**: each concern (data, model, training, serving, UI) lives in its own file/module.
- Keep it **configurable**: hyperparameters and paths live in a central config (e.g. `config.yaml` or `config.py`) — no magic numbers buried in code.
- Keep it **transparent**: log training metrics clearly, save checkpoints, and make evaluation easy to inspect.

---

## Machines

| Role | Machine | Responsibilities |
|---|---|---|
| Training / Inference Server | Main PC | Train model, host Flask API |
| Drawing Client | Touch screen laptop | Capture drawn digit, send to API, display result |

The two machines communicate over the local network via HTTP.

---

## Project Structure (Planned)

```
MNIST CNN/
├── PLAN.md                  ← this file
├── config.yaml              ← all hyperparameters and settings
├── requirements.txt
│
├── data/
│   └── (MNIST dataset downloaded here automatically)
│
├── model/
│   ├── __init__.py
│   └── cnn.py               ← CNN architecture definition
│
├── training/
│   ├── __init__.py
│   ├── dataset.py           ← data loading and transforms
│   ├── train.py             ← training loop
│   └── evaluate.py          ← evaluation / metrics
│
├── checkpoints/             ← saved model weights (.pth files)
│
├── server/
│   ├── __init__.py
│   ├── app.py               ← Flask app and /predict endpoint
│   └── inference.py         ← load model, preprocess, run inference
│
└── client/
    ├── (Option A) pygame_client.py   ← pygame drawing pad (runs on laptop)
    └── (Option B) index.html         ← HTML5 canvas drawing pad
        └── style.css
        └── app.js
```

> **Decision pending:** Drawing client — pygame vs. HTML5 canvas.
> - **pygame**: runs as a native app on the laptop, easier pixel-level control.
> - **HTML canvas**: runs in any browser, no install needed, works well with touch events.
> We can implement one or both. Leaning toward **HTML canvas** for touch screen friendliness.

---

## Phases

### Phase 1 — Model & Training (Main PC)

1. Define `config.yaml` with all settings (learning rate, batch size, epochs, etc.)
2. Build the CNN architecture in `model/cnn.py`
   - Input: 1×28×28 grayscale image
   - Conv layers → ReLU → MaxPool → Fully connected → Softmax (10 classes)
3. Set up data pipeline in `training/dataset.py`
   - Download MNIST via `torchvision.datasets`
   - Apply normalization transforms
4. Write training loop in `training/train.py`
   - Log loss and accuracy per epoch
   - Save best checkpoint to `checkpoints/`
5. Write `training/evaluate.py`
   - Report accuracy, confusion matrix on test set

### Phase 2 — Inference Server (Main PC)

1. Write `server/inference.py`
   - Load model from checkpoint
   - Preprocess raw image bytes → tensor
   - Return predicted digit + confidence scores
2. Write `server/app.py`
   - `POST /predict` — accepts a PNG/raw image, returns JSON `{ digit, confidence }`
   - `GET /health` — simple health check
3. Test with `curl` or Postman before connecting the client

### Phase 3 — Drawing Client (Touch Screen Laptop)

1. Implement drawing interface (HTML canvas preferred):
   - Finger/stylus draw digit on a dark canvas
   - "Clear" button to reset
   - Auto-send or "Submit" button to POST image to Flask server
   - Display returned digit and confidence prominently
2. Configure server IP/port in client so it can reach the main PC on the LAN

### Phase 4 — Polish & Transparency

- Add a training curve plot (loss/accuracy vs. epoch)
- Optionally display the top-3 predictions with a bar chart in the client UI
- Write a concise `README.md` with setup and usage instructions

---

## Key Configuration Values (to be finalized in `config.yaml`)

| Setting | Tentative Value |
|---|---|
| Learning rate | `1e-3` |
| Batch size | `64` |
| Epochs | `10` |
| Optimizer | Adam |
| Dropout | `0.25` |
| Image normalization | mean=`0.1307`, std=`0.3081` (MNIST standard) |
| Server host | `0.0.0.0` |
| Server port | `5000` |

---

## Open Questions / Decisions

- [ ] pygame vs. HTML canvas for drawing client?
- [ ] Auto-send on pen-lift vs. explicit submit button?
- [ ] Should the server run as a background service or be started manually?
- [ ] Do we want to support retraining from the client, or is inference-only enough?

---

## Future Updates

### 3D Model Visualization (Unreal Engine 5)

A real-time 3D visualization of the CNN as it performs inference, built in Unreal Engine 5.

**Concept:**
- Render the network layer-by-layer in a grid/matrix layout in 3D space
- Each neuron/weight is represented as a node; **weight magnitude** is shown via thickness or color (e.g. cool→hot color ramp)
- **Activation brightness** increases dynamically as a weight fires during a forward pass
- Visualization is **slowed down** — each layer activates sequentially with ~0.5s between layers, making the propagation human-readable even though the real forward pass takes milliseconds

**How it would connect:**
- The Flask server (or a new endpoint) streams per-layer activation data as JSON during inference
- A Python script or UE5 plugin (via HTTP polling or WebSocket) feeds that data into the UE5 scene in real time
- The drawing client triggers an inference → activations flow outward through the 3D scene layer by layer as the digit is being drawn

**Likely stack:**
- UE5 Blueprints and/or C++ for scene construction and animation
- Python script inside UE5 (via `unreal` Python API) or an external bridge process
- WebSocket or SSE (Server-Sent Events) for low-latency streaming from Flask to UE5

**Open questions for this feature:**
- [ ] Use UE5 Python API directly, or a separate bridge process?
- [ ] WebSocket vs. SSE for activation streaming?
- [ ] Pre-build the network graph at startup from model metadata, or generate procedurally?
- [ ] Support pausing/scrubbing through the activation playback?
