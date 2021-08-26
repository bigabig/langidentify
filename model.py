from torch import nn


class LangModel(nn.Module):
    def __init__(self):
        super(LangModel, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(667, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 6),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits
