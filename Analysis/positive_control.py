"""
Analysis/positive_control.py — positive control for the §6 protocol.
====================================================================
Self-contained (own CNN/MLP + MNIST loader + fixed-permutation ladder); NO imports
from or coupling to the RL stack. Implements the pre-registered design in
docs/POSITIVE_CONTROL_PREREGISTRATION.md:

  A = convolution, S = spatial locality. Ladder = a FIXED pixel permutation applied
  to a fraction q of positions (nested subsets, derangement within, identity outside
  -> a provable bijection at every rung; same relabeling for all images, train+test).
  q in {0, 0.25, 0.5, 0.75, 1.0}. CNN vs param-matched MLP, 10 seeds, 2-epoch budget.
  Metric: test accuracy. Predictions: CNN high->MLP as q->1; MLP flat; gap large->0.

Usage:
    python -m Analysis.positive_control run  --data-dir <dir> --out results/csv/positive_control.csv
    python -m Analysis.positive_control run  --smoke          # quick 1-seed sanity
    python -m Analysis.positive_control plot --out results/csv/positive_control.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

QS = (0.0, 0.25, 0.5, 0.75, 1.0)     # ladder rungs (scramble fraction)
PERM_SEED = 0                        # fixes the permutation regime (task definition)
N_SEEDS = 10                         # model/optimisation seeds (matches nav study)
EPOCHS = 2                           # fixed sub-asymptotic budget (pre-registered)
BATCH = 128
LR = 1e-3
PIX = 784                            # 28*28


# ---------------------------------------------------------------------------
# Fixed-permutation ladder (nested subsets + derangement -> bijection per rung)
# ---------------------------------------------------------------------------
def build_perms(perm_seed: int = PERM_SEED, qs=QS) -> dict[float, np.ndarray]:
    """One fixed nested permutation family. perms[q] is a length-784 index array with
    permuted_flat = flat[:, perms[q]] (i.e. output position j takes input sigma_q(j)).
    Identity outside the scrambled subset M_q; a permutation of M_q inside -> bijection."""
    rng = np.random.default_rng(perm_seed)
    order = rng.permutation(PIX)                 # priority order -> nested M_q
    perms = {}
    for q in qs:
        k = int(round(q * PIX))
        idx = order[:k].copy()                   # M_q (nested in k)
        shuffled = rng.permutation(idx)          # fixed permutation of M_q
        perm = np.arange(PIX)
        perm[idx] = shuffled                     # identity elsewhere
        assert len(np.unique(perm)) == PIX       # provable bijection
        perms[q] = perm
    return perms


# ---------------------------------------------------------------------------
# Models (param-matched CNN vs MLP)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):               # x: (N, 784)
        return self.net(x.view(-1, 1, 28, 28))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PIX, 124), nn.ReLU(), nn.Linear(124, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):               # x: (N, 784) — permutation-invariant by design
        return self.net(x)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mnist_flat(data_dir: str):
    """Return (Xtr, ytr, Xte, yte) as CPU tensors; X in [0,1] float32 shape (N,784)."""
    from torchvision import datasets
    tr = datasets.MNIST(data_dir, train=True, download=True)
    te = datasets.MNIST(data_dir, train=False, download=True)
    Xtr = (tr.data.float() / 255.0).view(-1, PIX)
    Xte = (te.data.float() / 255.0).view(-1, PIX)
    return Xtr, tr.targets.clone(), Xte, te.targets.clone()


# ---------------------------------------------------------------------------
# Train / eval one (arch, q, seed) cell
# ---------------------------------------------------------------------------
def train_eval(arch: str, perm: np.ndarray, data, seed: int,
               epochs: int = EPOCHS, device: str = "cpu") -> float:
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    pidx = torch.as_tensor(perm, dtype=torch.long)
    Xtr_p = Xtr.index_select(1, pidx).to(device)     # apply the fixed relabeling
    Xte_p = Xte.index_select(1, pidx).to(device)
    ytr_d, yte_d = ytr.to(device), yte.to(device)

    model = (CNN() if arch == "cnn" else MLP()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    n = Xtr_p.shape[0]
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr_p[b]), ytr_d[b]).backward()
            opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, Xte_p.shape[0], 1000):
            pred = model(Xte_p[i:i + 1000]).argmax(1)
            correct += (pred == yte_d[i:i + 1000]).sum().item()
    return 100.0 * correct / Xte_p.shape[0]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run(args) -> None:
    data = load_mnist_flat(args.data_dir)
    perms = build_perms()
    seeds = list(range(1 if args.smoke else args.seeds))
    qs = (0.0, 1.0) if args.smoke else QS
    epochs = 1 if args.smoke else args.epochs

    print(f"[pc] param counts: CNN={n_params(CNN()):,}  MLP={n_params(MLP()):,}", flush=True)
    ratio = n_params(CNN()) / n_params(MLP())
    assert 0.9 <= ratio <= 1.1, f"CNN/MLP param ratio {ratio:.3f} not within 10%"
    print(f"[pc] arch param ratio {ratio:.3f} (within 10%). "
          f"seeds={len(seeds)} qs={qs} epochs={epochs}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in ("cnn", "mlp"):
        for q in qs:
            for s in seeds:
                acc = train_eval(arch, perms[q], data, s, epochs=epochs)
                rows.append({"arch": arch, "q": q, "seed": s, "test_acc": round(acc, 4)})
                print(f"[pc] {arch} q={q} seed={s}: acc={acc:.2f}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "q", "seed", "test_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"[pc] wrote {len(rows)} rows -> {out}", flush=True)
    print("[pc] done.", flush=True)


# ---------------------------------------------------------------------------
# Analysis + figure
# ---------------------------------------------------------------------------
def _load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _iqm_ci(vals, reps=50000):
    from rliable import library as rly, metrics
    sd = {"x": np.asarray(vals, float).reshape(-1, 1)}
    fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    p, c = rly.get_interval_estimates(sd, fn, reps=reps)
    return float(p["x"][0]), float(c["x"][0, 0]), float(c["x"][1, 0])


def summarize(path):
    rows = _load_rows(path)
    out = {}
    for arch in ("cnn", "mlp"):
        for q in sorted({float(r["q"]) for r in rows}):
            vals = [float(r["test_acc"]) for r in rows
                    if r["arch"] == arch and float(r["q"]) == q]
            out[(arch, q)] = _iqm_ci(vals)
    return out, sorted({float(r["q"]) for r in rows})


def plot(args) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    summ, qs = summarize(args.out)
    fig_dir = Path(os.environ.get("ARCH_ABLATIONS_FIG_DIR", "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    cnn = np.array([summ[("cnn", q)] for q in qs])   # (Q,3): iqm,lo,hi
    mlp = np.array([summ[("mlp", q)] for q in qs])
    cC, cM = sns.color_palette("tab10")[0], sns.color_palette("tab10")[3]

    print("\n  q    CNN IQM (95% CI)        MLP IQM (95% CI)        gap")
    for i, q in enumerate(qs):
        print(f"  {q:<4} {cnn[i,0]:6.2f} [{cnn[i,1]:6.2f},{cnn[i,2]:6.2f}]  "
              f"{mlp[i,0]:6.2f} [{mlp[i,1]:6.2f},{mlp[i,2]:6.2f}]  {cnn[i,0]-mlp[i,0]:+.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(qs, mlp[:, 1], mlp[:, 2], color=cM, alpha=0.15)
    ax.fill_between(qs, cnn[:, 1], cnn[:, 2], color=cC, alpha=0.15)
    ax.plot(qs, mlp[:, 0], "-o", color=cM, lw=2, label="MLP")
    ax.plot(qs, cnn[:, 0], "-o", color=cC, lw=2, label="CNN")
    for i, q in enumerate(qs):                        # shrinking-gap annotations
        ax.annotate("", xy=(q, cnn[i, 0]), xytext=(q, mlp[i, 0]),
                    arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    ax.set_xlabel("scramble fraction  q   (spatial locality removed →)")
    ax.set_ylabel("test accuracy (IQM, 95% CI, 10 seeds)")
    ax.set_title("Positive control: the CNN–MLP gap tracks spatial locality (S)")
    ax.set_xticks(qs)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p = fig_dir / "pc_dose_response.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"\n[fig] {p}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Positive control (conv / spatial locality).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", type=str, default="data/mnist")
    r.add_argument("--out", type=str, default="results/csv/positive_control.csv")
    r.add_argument("--seeds", type=int, default=N_SEEDS)
    r.add_argument("--epochs", type=int, default=EPOCHS)
    r.add_argument("--smoke", action="store_true")
    r.set_defaults(func=run)
    p = sub.add_parser("plot")
    p.add_argument("--out", type=str, default="results/csv/positive_control.csv")
    p.set_defaults(func=plot)
    return ap


if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    a.func(a)
