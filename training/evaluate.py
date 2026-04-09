import os
import yaml
import torch

from model.cnn import MnistCNN
from training.dataset import get_loaders


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_loaders(cfg)

    model = MnistCNN(cfg).to(device)
    ckpt_path = os.path.join(cfg["checkpoints"]["dir"], cfg["checkpoints"]["filename"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    correct = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            for actual, predicted in zip(labels, preds):
                confusion[actual][predicted] += 1

    total = len(test_loader.dataset)
    acc = correct / total
    print(f"Test accuracy: {acc:.4f}  ({correct}/{total})\n")

    print("Confusion matrix (rows = actual, cols = predicted):")
    print("      " + "   ".join(str(i) for i in range(10)))
    print("    " + "─" * 43)
    for i, row in enumerate(confusion):
        print(f"  {i} | " + "   ".join(f"{v:3d}" for v in row))


if __name__ == "__main__":
    evaluate()
