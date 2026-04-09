import torch.nn as nn


class MnistCNN(nn.Module):
    """
    Two-layer CNN for MNIST digit classification.

    Architecture (driven entirely by config.yaml):
        Conv1 -> ReLU -> MaxPool(2x2)
        Conv2 -> ReLU -> MaxPool(2x2)
        Flatten -> FC1 -> ReLU -> Dropout -> FC2 (logits)

    With default 3x3 filters and same-padding the spatial size halves at each
    pool, so 28x28 -> 14x14 -> 7x7, giving a 64*7*7 = 3136-dim flat vector.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        k = m["kernel_size"]
        pad = k // 2  # 'same' padding: keeps spatial dims before pooling

        c1 = m["conv1_out_channels"]
        c2 = m["conv2_out_channels"]
        fc1 = m["fc1_out_features"]

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=k, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=k, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # After two 2x2 MaxPool layers: 28 -> 14 -> 7
        flat_features = c2 * 7 * 7

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, fc1),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(fc1, 10),  # 10 digit classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
