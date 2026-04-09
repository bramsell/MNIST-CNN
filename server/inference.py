import io
import os
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model.cnn import MnistCNN


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _preprocess(image: Image.Image, border_frac: float = 0.15) -> Image.Image:
    """
    1. Crop tightly to the bounding box of all non-black (drawn) pixels.
    2. Add a proportional border so the digit isn't edge-to-edge.
    3. Embed into a square canvas (centres the digit).
    4. Resize to 28x28 — this is the 'pixelize' step that matches MNIST.
    """
    bbox = image.getbbox()          # bounding box of non-zero pixels
    if bbox is None:                # blank canvas — return as-is
        return image.resize((28, 28), Image.LANCZOS)

    image = image.crop(bbox)
    w, h = image.size

    # Border proportional to the larger side
    pad = max(1, int(max(w, h) * border_frac))

    # Place onto a square black canvas with equal padding on all sides
    side = max(w, h) + 2 * pad
    square = Image.new("L", (side, side), 0)
    square.paste(image, ((side - w) // 2, (side - h) // 2))

    # Pixelize: downsample to 28x28 (MNIST resolution)
    return square.resize((28, 28), Image.LANCZOS)


# Final normalisation — applied after the manual preprocess step above
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _tensor_to_maps(t: torch.Tensor) -> list:
    """
    Convert a (C, H, W) activation tensor to a list of C flat arrays,
    each min-max normalised to [0, 1] for display.
    Returned values are rounded to 3dp to keep JSON compact.
    """
    maps = []
    for channel in t:                        # iterate over feature maps
        lo, hi = float(channel.min()), float(channel.max())
        span = hi - lo if hi != lo else 1.0
        normed = ((channel - lo) / span).tolist()
        # Each map is a list of rows (list of lists)
        maps.append([[round(v, 3) for v in row] for row in normed])
    return maps


def _tensor_to_values(t: torch.Tensor) -> list:
    """
    Convert a 1-D activation tensor to a min-max normalised float list.
    """
    lo, hi = float(t.min()), float(t.max())
    span = hi - lo if hi != lo else 1.0
    return [round(float((v - lo) / span), 3) for v in t]


class Predictor:
    """
    Loads the trained model once at startup, then accepts raw image bytes
    and returns the predicted digit, confidence scores, and (optionally)
    per-layer activation maps for visualisation.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MnistCNN(self.cfg).to(self.device)
        ckpt = self.cfg["checkpoints"]
        ckpt_path = os.path.join(ckpt["dir"], ckpt["filename"])
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        self.model = model

        self.capture_activations = self.cfg.get("visualization", {}).get(
            "capture_activations", True
        )

    def predict(self, image_bytes: bytes) -> dict:
        """
        Returns:
            {
                "digit": int,
                "confidence": float,
                "scores": [float x 10],
                "activations": [          # only when capture_activations=true
                    { "name": str, "type": "conv"|"pool"|"fc"|"output"|"input",
                      "shape": [...], "maps": [[...]] | "values": [...] },
                    ...
                ]
            }

        The `activations` list is ordered input → output and is a stable
        data contract — the same structure will be consumed by the web
        visualiser and, later, by the UE5 bridge.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = _preprocess(image)
        tensor = _transform(image).unsqueeze(0).to(self.device)   # (1,1,28,28)

        captured = {}

        if self.capture_activations:
            # ── Register forward hooks on named layers ──────────────────────
            # model.features indices:
            #   0=Conv2d  1=ReLU  2=MaxPool  3=Conv2d  4=ReLU  5=MaxPool
            # model.classifier indices:
            #   0=Flatten 1=Linear 2=ReLU 3=Dropout 4=Linear
            hook_specs = [
                ("conv1",  self.model.features[1]),   # after Conv1+ReLU
                ("pool1",  self.model.features[2]),   # after MaxPool1
                ("conv2",  self.model.features[4]),   # after Conv2+ReLU
                ("pool2",  self.model.features[5]),   # after MaxPool2
                ("fc1",    self.model.classifier[2]), # after FC1+ReLU
                ("output", self.model.classifier[4]), # raw FC2 logits
            ]

            hooks = []
            for name, layer in hook_specs:
                def _hook(mod, inp, out, n=name):
                    captured[n] = out.detach().cpu().squeeze(0)  # remove batch dim
                hooks.append(layer.register_forward_hook(_hook))

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze()

        if self.capture_activations:
            for h in hooks:
                h.remove()

        digit = int(probs.argmax())
        result = {
            "digit": digit,
            "confidence": round(float(probs[digit]), 4),
            "scores": [round(float(p), 4) for p in probs],
        }

        if self.capture_activations:
            # ── Also include the preprocessed input as the first layer ──────
            input_map = tensor.detach().cpu().squeeze(0)  # (1, 28, 28)

            activations = [
                {
                    "name": "input",
                    "type": "input",
                    "shape": list(input_map.shape),
                    "maps": _tensor_to_maps(input_map),
                }
            ]

            type_map = {
                "conv1": "conv", "pool1": "pool",
                "conv2": "conv", "pool2": "pool",
                "fc1":   "fc",   "output": "output",
            }

            for name, _ in hook_specs:
                act = captured[name]
                if act.dim() == 3:          # (C, H, W) — spatial feature map
                    activations.append({
                        "name": name,
                        "type": type_map[name],
                        "shape": list(act.shape),
                        "maps": _tensor_to_maps(act),
                    })
                else:                       # (N,) — flat FC layer
                    activations.append({
                        "name": name,
                        "type": type_map[name],
                        "shape": list(act.shape),
                        "values": _tensor_to_values(act),
                    })

            result["activations"] = activations

        return result
