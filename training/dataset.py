from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) configured from cfg."""
    d = cfg["data"]
    t = cfg["training"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((d["mean"],), (d["std"],)),
    ])

    train_set = datasets.MNIST(
        root=d["dir"], train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=d["dir"], train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=t["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=t["batch_size"], shuffle=False, num_workers=0
    )

    return train_loader, test_loader
