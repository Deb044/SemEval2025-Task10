"""Two-layer label GCN using plain PyTorch (no graph framework)."""
import copy

import numpy as np
import torch
from torch import nn

from .features import get_encoder


class LabelGCN(nn.Module):
    def __init__(self, article_dim, initial, adjacency, output_indices):
        super().__init__()
        self.register_buffer("initial", initial)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("output_indices", output_indices)
        self.layer1 = nn.Linear(initial.shape[1], 128)
        self.layer2 = nn.Linear(128, 128)
        self.article = nn.Linear(article_dim, 128)
        self.bias = nn.Parameter(torch.zeros(len(output_indices)))
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        labels = torch.relu(self.layer1(self.adjacency @ self.initial))
        labels = self.layer2(self.adjacency @ self.dropout(labels))
        return self.article(x) @ labels[self.output_indices].T / np.sqrt(128) + self.bias


def fit_graph(x, y, xd, yd, labels, taxonomy, seed, ablation):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    nodes = [("fine", label) for label in labels] + [("parent", p) for p in sorted({v["parent"] for v in taxonomy.values()})]
    lookup = {node: i for i, node in enumerate(nodes)}
    initial = get_encoder().encode([taxonomy[label]["definition"] + " " + label if kind == "fine" else label
                                     for kind, label in nodes])
    if ablation == "random-labels":
        initial = np.random.default_rng(seed).normal(size=initial.shape).astype("float32")
        initial /= np.linalg.norm(initial, axis=1, keepdims=True)
    adjacency = np.eye(len(nodes), dtype=np.float32)
    if ablation != "self-loops":
        for label in labels:
            a, b = lookup[("fine", label)], lookup[("parent", taxonomy[label]["parent"])]
            adjacency[a, b] = adjacency[b, a] = 1
    degree = adjacency.sum(axis=1) ** -.5
    adjacency = degree[:, None] * adjacency * degree[None, :]
    model = LabelGCN(x.shape[1], torch.tensor(initial), torch.tensor(adjacency), torch.arange(len(labels)))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, y, xd, yd = [torch.tensor(a, dtype=torch.float32) for a in (x, y, xd, yd)]
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, stale, best_state = float("inf"), 0, None
    for _ in range(100):
        model.train()
        for indices in torch.randperm(len(x)).split(32):
            optimizer.zero_grad()
            loss_fn(model(x[indices]), y[indices]).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = loss_fn(model(xd), yd).item()
        if dev_loss < best_loss:
            best_loss, stale, best_state = dev_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    return model, graph_probabilities(model, xd.numpy())


def graph_probabilities(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
