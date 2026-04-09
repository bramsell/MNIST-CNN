import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from model.cnn import MnistCNN
from training.dataset import get_loaders


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    t = cfg["training"]
    ckpt_dir = cfg["checkpoints"]["dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(t["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(cfg)

    model = MnistCNN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    ckpt_path = os.path.join(ckpt_dir, cfg["checkpoints"]["filename"])

    for epoch in range(1, t["epochs"] + 1):

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{t['epochs']}", unit="batch", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()

        acc = correct / len(test_loader.dataset)
        print(f"Epoch {epoch:>2}/{t['epochs']}  loss={avg_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"           -> checkpoint saved  (best val_acc={best_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    train()
