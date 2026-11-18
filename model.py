import torch, torch.nn as nn, torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.MaxPool2d(2))


class SiameseNet(nn.Module):
    """One shared CNN branch -> L2-normalised 128-d embedding. Distance is computed outside."""
    def __init__(self, emb=128):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256),
                                      nn.AdaptiveAvgPool2d((4, 8)))   # keeps spatial layout
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256 * 4 * 8, 256),
                                nn.ReLU(True), nn.Linear(256, emb))

    def embed(self, x):
        return F.normalize(self.fc(self.features(x)), dim=1)

    def forward(self, a, b):                      # both go through the SAME weights
        ea, eb = self.embed(a), self.embed(b)
        return (ea - eb).pow(2).sum(1).add(1e-9).sqrt()


def contrastive_loss(d, y, margin=1.0):
    """y=1 (match): pull together; y=0: push apart until distance >= margin."""
    return (y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)).mean()
