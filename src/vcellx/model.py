"""vcellx · model and training.

One backbone, three conditioning variants. The *only* thing that changes
between variants is how the epigenomic signal reaches the perturbation
conditioning vector:

  graph   z = emb[p] + W · (A_norm E)[p]      signal diffused over the GRN
  shuffle z = emb[p] + W · (A_shuf E)[p]      same degree, randomised wiring
  direct  z = emb[p] + W · E[p]               target gene's own features only

`emb` is a learned per-target embedding shared by all three, so the comparison
does not conflate "has an embedding" with "has a graph".

Loss: MMD between the predicted cell set and the observed perturbed cell set.
The sets are unpaired by construction (different cells), which is exactly the
situation in single-cell perturbation data.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, dropout=0.0, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class VCellX(nn.Module):
    def __init__(self, n_genes, n_targets, epi, adj=None, d=256, arm="direct",
                 dropout=0.1, hidden=512):
        super().__init__()
        self.arm = arm
        self.n_genes = n_genes
        self.d = d

        self.enc = mlp([n_genes, hidden, d], dropout)
        self.dec = mlp([d, hidden, n_genes], dropout)
        self.emb = nn.Embedding(n_targets, d)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.use_epi = arm in ("graph", "shuffle", "direct")
        if self.use_epi:
            E = torch.as_tensor(epi, dtype=torch.float32)          # (G, k)
            k = E.shape[1]
            self.register_buffer("epi", E)
            if arm in ("graph", "shuffle"):
                assert adj is not None, "graph/shuffle arm needs an adjacency"
                A = torch.as_tensor(adj, dtype=torch.float32)
                deg = A.sum(1, keepdim=True).clamp(min=1e-6)
                self.register_buffer("adj_norm", A / deg)           # row-normalised
                self.epi_proj = nn.Linear(k, d, bias=False)
            else:
                self.epi_proj = nn.Linear(k, d, bias=False)

    def condition(self, pid):
        z = self.emb(pid)                                          # (B, d)
        if not self.use_epi:
            return z
        if self.arm in ("graph", "shuffle"):
            H = self.adj_norm @ self.epi                           # (G, k) diffuse
            g = H[pid]
        else:
            g = self.epi[pid]
        return z + self.epi_proj(g)

    def forward(self, x, pid):
        """x: (B, S, G) control cells -> (B, S, G) predicted perturbed cells."""
        B, S, G = x.shape
        h = self.enc(x.reshape(B * S, G)).reshape(B, S, self.d)
        z = self.condition(pid).unsqueeze(1)                       # (B, 1, d)
        out = self.dec(h + z)
        return out


# --------------------------------------------------------------------------
# MMD loss (energy-distance kernel) on sets
# --------------------------------------------------------------------------
def mmd2_energy(pred, tgt):
    """pred, tgt: (B, S, G). Mean over the batch of the squared energy distance.

    D = 2 E||x-y|| - E||x-x'|| - E||y-y'||, computed per set and averaged.
    """
    pw_pp = torch.cdist(pred, pred)          # (B, S, S)
    pw_tt = torch.cdist(tgt, tgt)
    pw_pt = torch.cdist(pred, tgt)
    S = pred.shape[1]
    off = 1.0 - torch.eye(S, device=pred.device)                   # exclude self terms
    n_off = off.sum()
    a = (pw_pp * off).sum((1, 2)) / n_off
    b = (pw_tt * off).sum((1, 2)) / n_off
    c = pw_pt.mean((1, 2))
    return (2 * c - a - b).mean()


def mmd2_mmd(pred, tgt):
    """Standard biased MMD^2 with the energy kernel k(a,b) = -||a-b||."""
    return -mmd2_energy(pred, tgt) + 0.0   # same quantity, opposite sign convention


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_model(model, batches, epochs=30, lr=1e-3, wd=1e-4, bs=8,
                device="cpu", log_every=20, seed=0, target_index=None):
    """batches: SetBatch. target_index maps target gene -> integer id."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ctrl = torch.as_tensor(batches.ctrl, dtype=torch.float32)
    pert = torch.as_tensor(batches.pert, dtype=torch.float32)
    pid = torch.as_tensor([target_index[t] for t in batches.tgt_label], dtype=torch.long)
    N = ctrl.shape[0]

    hist = []
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(N)
        run = 0.0
        nb = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            x = ctrl[idx].to(device)
            y = pert[idx].to(device)
            p = pid[idx].to(device)
            pred = model(x, p)
            loss = mmd2_energy(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
            step += 1
            if log_every and step % log_every == 0:
                print(f"    ep{ep+1} step{step} mmd2 {run/nb:.4f}", flush=True)
        sched.step()
        hist.append(run / max(nb, 1))
        print(f"  epoch {ep+1}/{epochs}  mmd2 {hist[-1]:.5f}", flush=True)
    return hist


@torch.no_grad()
def predict_mean(model, ctrl: np.ndarray, pid_int: int, device="cpu", chunk=64):
    """Predict the mean expression of the perturbed population for one target.

    Returns (G,) mean prediction over the control cells supplied.
    """
    model.eval()
    B = ctrl.shape[0]
    outs = []
    for i in range(0, B, chunk):
        x = torch.as_tensor(ctrl[i:i + chunk], dtype=torch.float32).to(device)
        p = torch.full((x.shape[0],), pid_int, dtype=torch.long, device=device)
        outs.append(model(x, p).mean(1).cpu().numpy())
    return np.concatenate(outs, 0).mean(0)
