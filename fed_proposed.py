#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fed_proposed_v3.py

A safer PD-only federated learning baseline designed to:
1) improve weak clients via transfer from stronger clients,
2) protect strong clients from negative transfer,
3) prevent unstable clients from contaminating others,
4) keep per-round logs for downstream analysis.
    
python fed_qfedavg_pd.py \
  --pd_clients EN ZH ES IT CZ \
  --seed 42 \
  --rounds 100 \
  --q 5 \
  --local_lr 0.01 \
  --local_epochs 3 \
  --outdir results/qfedavg_seed42
  
  
for seed in 42 52 62
do
  python fed_qfedavg_pd.py \
    --pd_clients EN ZH ES IT CZ \
    --seed ${seed} \
    --rounds 100 \
    --q 5 \
    --local_lr 0.01 \
    --local_epochs 3 \
    --outdir results/qfedavg_seed${seed}
done
"""

import os
import glob
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef

torch.backends.cudnn.enabled = False


@dataclass
class CFG:
    pd_en_train_root: str = "data/PD/English/Train/"
    pd_en_test_root: str = "data/PD/English/Test/"
    pd_en_train_precomp_mel: str = "data/PD/English/precomputed_TS_mel/Train"
    pd_en_test_precomp_mel: str = "data/PD/English/precomputed_TS_mel/Test"

    pd_zh_train_root: str = "data/PD/Chinese/Train/"
    pd_zh_test_root: str = "data/PD/Chinese/Test/"
    pd_zh_train_precomp_mel: str = "data/PD/Chinese/precomputed_TS_mel/Train"
    pd_zh_test_precomp_mel: str = "data/PD/Chinese/precomputed_TS_mel/Test"

    pd_es_train_root: str = "data/PD/Spanish/Train/"
    pd_es_test_root: str = "data/PD/Spanish/Test/"
    pd_es_train_precomp_mel: str = "data/PD/Spanish/precomputed_TS_mel/Train"
    pd_es_test_precomp_mel: str = "data/PD/Spanish/precomputed_TS_mel/Test"

    pd_it_train_root: str = "data/PD/Italian/Train/"
    pd_it_test_root: str = "data/PD/Italian/Test/"
    pd_it_train_precomp_mel: str = "data/PD/Italian/precomputed_TS_mel/Train"
    pd_it_test_precomp_mel: str = "data/PD/Italian/precomputed_TS_mel/Test"

    pd_cz_train_root: str = "data/PD/Czech/Train/"
    pd_cz_test_root: str = "data/PD/Czech/Test/"
    pd_cz_train_precomp_mel: str = "data/PD/Czech/precomputed_TS_mel/Train"
    pd_cz_test_precomp_mel: str = "data/PD/Czech/precomputed_TS_mel/Test"

    ts_max_len_mel: int = 120
    batch_size: int = 8
    rounds: int = 100

    en_epochs_local: int = 3
    zh_epochs_local: int = 3
    es_epochs_local: int = 3
    it_epochs_local: int = 3
    cz_epochs_local: int = 3

    lr_local: float = 2e-3
    lr_private_factor: float = 0.2
    weight_decay: float = 5e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    smooth_w: float = 0.85
    min_w: float = 0.05
    max_w: float = 0.70
    weight_floor_add: float = 0.05

    safety_history: int = 5
    safety_loss_tau: float = 0.04
    safety_f1_tau: float = 0.025
    safety_floor: float = 0.20

    hard_rollback_loss_tol: float = 0.02
    hard_rollback_f1_tol: float = 0.015
    donor_ban_rounds: int = 2

    f1_weak_th: float = 0.55
    f1_strong_th: float = 0.85

    alpha_min: float = 0.10
    alpha_mid: float = 0.40
    alpha_max: float = 0.70

    donor_cos_thr: float = 0.05
    donor_min_scale: float = 0.05
    donor_pow: float = 2.0

    global_ema_beta: float = 0.90

    mu_prox: float = 1e-3

    f1_ref: float = 0.60
    max_gamma: float = 2.0

    personalization_epochs: int = 5
    lr_personal: float = 5e-4
    lambda_l2sp: float = 1e-4

    base_smoothing: float = 0.03
    max_grad_norm: float = 1.0


cfg = CFG()
ALL_PD = ["EN", "ZH", "ES", "IT", "CZ"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _ensure_nonempty(name: str, n: int, hint: str):
    if n <= 0:
        raise RuntimeError(f"{name} is empty. {hint}")


def snapshot_state_dict(sd: Dict[str, torch.Tensor], device=None, to_cpu: bool = False) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        t = v.detach().clone()
        if to_cpu:
            t = t.cpu()
        elif device is not None:
            t = t.to(device)
        out[k] = t
    return out


def weighted_average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: np.ndarray, device) -> Dict[str, torch.Tensor]:
    keys = list(state_dicts[0].keys())
    out = {}
    for k in keys:
        acc = None
        for sd, w in zip(state_dicts, weights.tolist()):
            v = sd[k].detach().to(device)
            acc = v * w if acc is None else acc + v * w
        out[k] = acc
    return out


def ema_state_dict(old_sd: Dict[str, torch.Tensor], new_sd: Dict[str, torch.Tensor], beta: float, device) -> Dict[str, torch.Tensor]:
    return {k: beta * old_sd[k].detach().to(device) + (1.0 - beta) * new_sd[k].detach().to(device) for k in old_sd.keys()}


def _load_ts(root: str, base: str) -> np.ndarray:
    p = os.path.join(root, base + ".npy")
    arr = np.load(p)
    if arr.ndim == 3:
        T, F, C = arr.shape
        arr = arr.reshape(T, F * C)
    return arr.astype(np.float32)


def pad_or_crop(arr: np.ndarray, max_len: int) -> Tuple[np.ndarray, int]:
    T = arr.shape[0]
    if T > max_len:
        return arr[:max_len], max_len
    if T == max_len:
        return arr, T
    pad = np.tile(arr[-1:, :], (max_len - T, 1))
    return np.concatenate([arr, pad], 0), T


def scan_pd_en_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "_pd_" in b or b.endswith("_pd.wav") or "_parkinson" in b:
            X.append(p); y.append(1)
        elif "_hc_" in b or b.endswith("_hc.wav") or "_control" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No English PD/HC .wav found under {root_dir}")
    return X, np.array(y, dtype=np.int64)


def scan_pd_zh_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if b.startswith("pd"):
            X.append(p); y.append(1)
        elif b.startswith("hc"):
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Chinese PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)


def scan_pd_es_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        name = os.path.basename(p).upper()
        if "AVPEPUDEAC" in name:
            X.append(p); y.append(0)
        elif "AVPEPUDEA" in name:
            X.append(p); y.append(1)
    if not X:
        raise FileNotFoundError(f"No Spanish PD/HC wav under {root_dir} (AVPEPUDEA / AVPEPUDEAC expected)")
    return X, np.array(y, dtype=np.int64)


def scan_pd_it_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "_pd" in b:
            X.append(p); y.append(1)
        elif "_hc" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Italian PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)


def scan_pd_cz_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "pd" in b:
            X.append(p); y.append(1)
        elif "hc" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Czech PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)


PD_SCAN = {"EN": scan_pd_en_audio, "ZH": scan_pd_zh_audio, "ES": scan_pd_es_audio, "IT": scan_pd_it_audio, "CZ": scan_pd_cz_audio}
PD_PATHS = {
    "EN": ("pd_en_train_root", "pd_en_test_root", "pd_en_train_precomp_mel", "pd_en_test_precomp_mel"),
    "ZH": ("pd_zh_train_root", "pd_zh_test_root", "pd_zh_train_precomp_mel", "pd_zh_test_precomp_mel"),
    "ES": ("pd_es_train_root", "pd_es_test_root", "pd_es_train_precomp_mel", "pd_es_test_precomp_mel"),
    "IT": ("pd_it_train_root", "pd_it_test_root", "pd_it_train_precomp_mel", "pd_it_test_precomp_mel"),
    "CZ": ("pd_cz_train_root", "pd_cz_test_root", "pd_cz_train_precomp_mel", "pd_cz_test_precomp_mel"),
}
PD_EPOCHS = {"EN": "en_epochs_local", "ZH": "zh_epochs_local", "ES": "es_epochs_local", "IT": "it_epochs_local", "CZ": "cz_epochs_local"}


class PDMelDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, mel_root: str, max_len: int, train: bool = True):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mel_root = mel_root
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        base = os.path.splitext(os.path.basename(p))[0]
        arr = _load_ts(self.mel_root, base)
        if self.train:
            T = arr.shape[0]
            if T > self.max_len:
                s = np.random.randint(0, T - self.max_len + 1)
                arr = arr[s:s + self.max_len]
                L = self.max_len
            else:
                arr, L = pad_or_crop(arr, self.max_len)
        else:
            arr, L = pad_or_crop(arr, self.max_len)
        return torch.from_numpy(arr).float(), L, self.labels[i], i


def pd_collate(batch):
    xs, ls, ys, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ls, dtype=torch.long), torch.stack(ys, 0), torch.tensor(idxs, dtype=torch.long)


class SharedMelEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 192, layers: int = 1, dropout: float = 0.35):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hid * 4

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        _, T, _ = h.shape
        mask = torch.arange(T, device=h.device)[None, :] < lengths[:, None]
        m = mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp_min(1.0)
        mean_pool = (h * m).sum(1) / denom
        h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool, _ = h_masked.max(1)
        return torch.cat([mean_pool, max_pool], 1)


class PDClientModel(nn.Module):
    def __init__(self, shared_mel: SharedMelEncoder, num_classes: int = 2):
        super().__init__()
        self.shared_mel = shared_mel
        self.adapter = nn.Sequential(nn.LayerNorm(self.shared_mel.out_dim), nn.Linear(self.shared_mel.out_dim, self.shared_mel.out_dim), nn.ReLU(True))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.shared_mel.out_dim, 256), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, mel_x: torch.Tensor, mel_L: torch.Tensor):
        feat = self.shared_mel(mel_x, mel_L)
        feat = self.adapter(feat)
        return self.head(feat)


def make_class_weights(y: np.ndarray) -> torch.Tensor:
    cls, cnt = np.unique(y, return_counts=True)
    num_classes = len(cls)
    tot = float(cnt.sum())
    w = np.ones(num_classes, dtype=np.float32)
    for c, k in zip(cls, cnt):
        w[int(c)] = tot / (num_classes * float(k))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_pd_dl(model: PDClientModel, dl: DataLoader, device) -> Tuple[float, float, float]:
    model.to(device).eval()
    ce = nn.CrossEntropyLoss()
    ys, ps, losses = [], [], []
    for X, L, y, _ in dl:
        X, L, y = X.to(device), L.to(device), y.to(device)
        logits = model(X, L)
        loss = ce(logits, y)
        losses.append(loss.item())
        ps += logits.argmax(1).cpu().tolist()
        ys += y.cpu().tolist()
    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    return float(np.mean(losses)) if losses else 0.0, float(acc), float(f1)


@torch.no_grad()
def eval_pd_multicrop(model: PDClientModel, paths: List[str], labels: np.ndarray, mel_root: str, device, max_len: int, n_crops: int = 3) -> Tuple[float, float]:
    model.to(device).eval()
    ys, ps = [], []
    for p, y in zip(paths, labels):
        base = os.path.splitext(os.path.basename(p))[0]
        mel_arr = _load_ts(mel_root, base)
        mel_t = torch.from_numpy(mel_arr).float()

        def make_crops(arr_t: torch.Tensor, target_len: int):
            T = arr_t.size(0)
            if T <= target_len:
                pad = arr_t[-1:, :].repeat(target_len - T, 1)
                return [torch.cat([arr_t, pad], 0)]
            gap = (T - target_len) // (n_crops - 1) if n_crops > 1 else 0
            crops = []
            for i in range(n_crops):
                s = i * gap
                e = s + target_len
                if e > T:
                    s = T - target_len
                    e = T
                crops.append(arr_t[s:e])
            return crops

        crops = make_crops(mel_t, max_len)
        logits_agg = 0.0
        for c in crops:
            L = torch.tensor([c.size(0)], dtype=torch.long, device=device)
            X = c.unsqueeze(0).to(device)
            logits = model(X, L)
            logits_agg = logits_agg + logits
        logits_agg = logits_agg / len(crops)
        pred = logits_agg.argmax(1).item()
        ys.append(int(y))
        ps.append(pred)

    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    return float(acc), float(f1)




def binary_classification_metrics(y_true, y_pred):
    """Return accuracy plus class-balance-aware metrics for binary PD/HC classification.

    Label convention: HC=0, PD=1.
    sensitivity = PD recall; specificity = HC recall.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return {
            "acc": 0.0, "bal_acc": 0.0, "f1": 0.0, "weighted_f1": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0,
            "sensitivity": 0.0, "specificity": 0.0, "mcc": 0.0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0,
    }


@torch.no_grad()
def eval_pd_dl_full(model: PDClientModel, dl: DataLoader, device) -> Dict[str, float]:
    model.to(device).eval()
    ce = nn.CrossEntropyLoss()
    ys, ps, losses = [], [], []
    for X, L, y, _ in dl:
        X, L, y = X.to(device), L.to(device), y.to(device)
        logits = model(X, L)
        loss = ce(logits, y)
        losses.append(float(loss.item()))
        ps += logits.argmax(1).cpu().tolist()
        ys += y.cpu().tolist()
    out = binary_classification_metrics(ys, ps)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


@torch.no_grad()
def eval_pd_multicrop_full(model: PDClientModel, paths: List[str], labels: np.ndarray, mel_root: str, device, max_len: int, n_crops: int = 3) -> Dict[str, float]:
    model.to(device).eval()
    ys, ps = [], []
    for p, y in zip(paths, labels):
        base = os.path.splitext(os.path.basename(p))[0]
        mel_arr = _load_ts(mel_root, base)
        mel_t = torch.from_numpy(mel_arr).float()

        def make_crops(arr_t: torch.Tensor, target_len: int):
            T = arr_t.size(0)
            if T <= target_len:
                pad = arr_t[-1:, :].repeat(target_len - T, 1)
                return [torch.cat([arr_t, pad], 0)]
            gap = (T - target_len) // (n_crops - 1) if n_crops > 1 else 0
            crops = []
            for i in range(n_crops):
                s = i * gap
                e = s + target_len
                if e > T:
                    s = T - target_len
                    e = T
                crops.append(arr_t[s:e])
            return crops

        crops = make_crops(mel_t, max_len)
        logits_agg = 0.0
        for c in crops:
            L = torch.tensor([c.size(0)], dtype=torch.long, device=device)
            X = c.unsqueeze(0).to(device)
            logits = model(X, L)
            logits_agg = logits_agg + logits
        logits_agg = logits_agg / len(crops)
        pred = logits_agg.argmax(1).item()
        ys.append(int(y))
        ps.append(pred)
    return binary_classification_metrics(ys, ps)

def compute_gamma_for_client(cid: str, perf_hist: Dict[str, List[Dict]], f1_ref: float, max_gamma: float, ema_beta: float = 0.6, window: int = 5) -> float:
    hist = perf_hist.get(cid, [])
    if not hist:
        return 0.0
    f1_vals = [float(h["val_f1"]) for h in hist[-window:]]
    if not f1_vals:
        return 0.0
    ema_f1 = f1_vals[0]
    for v in f1_vals[1:]:
        ema_f1 = ema_beta * v + (1.0 - ema_beta) * ema_f1
    if ema_f1 <= f1_ref:
        return 0.0
    denom = max(1.0 - f1_ref, 1e-8)
    t = (ema_f1 - f1_ref) / denom
    t = max(0.0, min(1.0, t))
    return float(max_gamma * t)


def prox_term(shared: SharedMelEncoder, ref_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    reg = 0.0
    for k, v in shared.state_dict().items():
        if k in ref_sd:
            reg = reg + (v - ref_sd[k]).pow(2).sum()
    return reg


def capture_private_params(model: nn.Module, shared_prefix: str = "shared_mel") -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if shared_prefix in n:
            continue
        out[n] = p.detach().clone()
    return out


def l2sp_private(model: nn.Module, init_private: Dict[str, torch.Tensor], device, shared_prefix: str = "shared_mel") -> torch.Tensor:
    reg = 0.0
    for n, p in model.named_parameters():
        if shared_prefix in n:
            continue
        if n in init_private:
            reg = reg + (p - init_private[n].to(device)).pow(2).sum()
    return reg


def train_pd_local(model: PDClientModel, dl: DataLoader, device, epochs: int, lr_shared: float, lr_private: float,
                   class_w: torch.Tensor, cid: str, perf_hist: Dict[str, List[Dict]], use_focal: bool = True,
                   ref_shared_sd: Optional[Dict[str, torch.Tensor]] = None, mu_prox: float = 0.0,
                   update_shared: bool = True, update_private: bool = True, base_smoothing: float = 0.03,
                   max_grad_norm: float = 1.0, l2sp_init_private: Optional[Dict[str, torch.Tensor]] = None,
                   lambda_l2sp: float = 0.0) -> Tuple[PDClientModel, float]:
    model.to(device).train()
    class_w = class_w.to(device)
    gamma = compute_gamma_for_client(cid, perf_hist, cfg.f1_ref, cfg.max_gamma) if use_focal else 0.0
    params_shared = list(model.shared_mel.parameters()) if update_shared else []
    params_private = list(model.adapter.parameters()) + list(model.head.parameters()) if update_private else []
    param_groups = []
    if params_shared:
        param_groups.append({"params": params_shared, "lr": lr_shared})
    if params_private:
        param_groups.append({"params": params_private, "lr": lr_private})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    all_losses = []
    for _ in range(epochs):
        for X, L, y, _ in dl:
            X, L, y = X.to(device), L.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X, L)
            if gamma <= 1e-6:
                loss = nn.CrossEntropyLoss(weight=class_w, label_smoothing=base_smoothing)(logits, y)
            else:
                ce = F.cross_entropy(logits, y, weight=class_w, label_smoothing=base_smoothing, reduction="none")
                with torch.no_grad():
                    prob = F.softmax(logits, dim=1)
                    p_t = prob[torch.arange(prob.size(0)), y]
                loss = ((1.0 - p_t).pow(gamma) * ce).mean()
            if (mu_prox > 0.0) and (ref_shared_sd is not None) and update_shared:
                loss = loss + mu_prox * prox_term(model.shared_mel, ref_shared_sd)
            if (lambda_l2sp > 0.0) and (l2sp_init_private is not None) and update_private:
                loss = loss + lambda_l2sp * l2sp_private(model, l2sp_init_private, device=device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            all_losses.append(loss.item())
    return model, float(np.mean(all_losses)) if all_losses else 0.0


def is_strong_client(f1: float, cfg: CFG) -> bool:
    return float(f1) > cfg.f1_strong_th


def classify_client_role(cid: str, perf_hist: Dict[str, List[Dict]], cfg: CFG) -> str:
    hist = perf_hist.get(cid, [])
    if not hist:
        return "weak"
    f1 = float(hist[-1]["val_f1"])
    if f1 > cfg.f1_strong_th:
        return "strong"
    if f1 < cfg.f1_weak_th:
        return "weak"
    return "mid"


def compute_alpha_global(f1: float, loss: float, cfg: CFG) -> float:
    if f1 < cfg.f1_weak_th:
        return cfg.alpha_max
    if f1 > cfg.f1_strong_th:
        return cfg.alpha_min
    return cfg.alpha_mid


def compute_safety_score(delta_f1: float, delta_loss: float, cfg: CFG) -> float:
    raw = 1.0
    if delta_f1 < 0:
        raw *= math.exp(delta_f1 / max(cfg.safety_f1_tau, 1e-8))
    if delta_loss > 0:
        raw *= math.exp(-delta_loss / max(cfg.safety_loss_tau, 1e-8))
    return float(np.clip(cfg.safety_floor + (1.0 - cfg.safety_floor) * raw, 0.0, 1.0))


def compute_client_safety_gate(cid: str, perf_hist: Dict[str, List[Dict]], cfg: CFG) -> float:
    hist = perf_hist.get(cid, [])
    if len(hist) < 2:
        return 1.0
    recent = hist[-cfg.safety_history:]
    losses = [float(h["val_loss"]) for h in recent]
    f1s = [float(h["val_f1"]) for h in recent]
    last_loss = losses[-1]
    best_prev_loss = min(losses[:-1]) if len(losses) > 1 else last_loss
    delta_loss = last_loss - best_prev_loss
    last_f1 = f1s[-1]
    best_prev_f1 = max(f1s[:-1]) if len(f1s) > 1 else last_f1
    delta_f1 = last_f1 - best_prev_f1
    return compute_safety_score(delta_f1=delta_f1, delta_loss=delta_loss, cfg=cfg)


def hard_safety_violated(cid: str, perf_hist: Dict[str, List[Dict]], cfg: CFG) -> bool:
    hist = perf_hist.get(cid, [])
    if len(hist) < 2:
        return False
    last = hist[-1]
    prev = hist[-2]
    return ((float(last["val_loss"]) - float(prev["val_loss"])) > cfg.hard_rollback_loss_tol) or ((float(prev["val_f1"]) - float(last["val_f1"])) > cfg.hard_rollback_f1_tol)


def compute_client_weight(val_loss: float, val_f1: float, size: int, safety: float, cfg: CFG) -> float:
    w = math.exp(-float(val_loss)) * (0.5 + float(max(val_f1, 0.0)))
    w = max(cfg.min_w, min(cfg.max_w, w))
    return float(w * math.sqrt(max(int(size), 1)) * safety)


def flatten_delta_from_sd(new_sd: Dict[str, torch.Tensor], ref_sd: Dict[str, torch.Tensor], device) -> torch.Tensor:
    vecs = []
    for k in new_sd.keys():
        if k in ref_sd:
            vecs.append((new_sd[k].detach().to(device) - ref_sd[k].detach().to(device)).reshape(-1))
    return torch.cat(vecs, dim=0) if vecs else torch.zeros(1, device=device)


def cosine_delta(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor], ref_sd: Dict[str, torch.Tensor], device) -> float:
    da = flatten_delta_from_sd(sd_a, ref_sd, device)
    db = flatten_delta_from_sd(sd_b, ref_sd, device)
    na = float(torch.norm(da, p=2).item())
    nb = float(torch.norm(db, p=2).item())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(F.cosine_similarity(da, db, dim=0).item())


def donor_scale_from_cos(cos_val: float, cfg: CFG) -> float:
    if cos_val >= cfg.donor_cos_thr:
        return 1.0
    ratio = (cos_val + 1.0) / max(cfg.donor_cos_thr + 1.0, 1e-8)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(cfg.donor_min_scale + (1.0 - cfg.donor_min_scale) * (ratio ** cfg.donor_pow))


def compute_global_weights(pd_clients: List[str], perf_hist: Dict[str, List[Dict]], dataset_sizes: Dict[str, int], prev_w: Optional[np.ndarray], donor_ban_counter: Dict[str, int], cfg: CFG):
    gates, roles, raw = {}, {}, []
    for cid in pd_clients:
        gates[cid] = compute_client_safety_gate(cid, perf_hist, cfg)
        roles[cid] = classify_client_role(cid, perf_hist, cfg)
        if donor_ban_counter.get(cid, 0) > 0 or not perf_hist.get(cid):
            raw.append(1e-8)
            continue
        last = perf_hist[cid][-1]
        raw.append(compute_client_weight(last["val_loss"], last["val_f1"], dataset_sizes.get(cid, 1), gates[cid], cfg))
    w = np.array(raw, dtype=np.float64)
    w = np.maximum(w, 1e-8)
    w = w + cfg.weight_floor_add
    w = w / w.sum()
    if prev_w is not None:
        w = cfg.smooth_w * prev_w + (1.0 - cfg.smooth_w) * w
        w = w / w.sum()
    return w, gates, roles


def build_personalized_server_state(target_cid: str, pd_clients: List[str], client_sds: Dict[str, Dict[str, torch.Tensor]],
                                    ref_sd: Dict[str, torch.Tensor], perf_hist: Dict[str, List[Dict]],
                                    dataset_sizes: Dict[str, int], roles: Dict[str, str], donor_ban_counter: Dict[str, int],
                                    device, cfg: CFG):
    target_role = roles[target_cid]
    target_sd = client_sds[target_cid]
    donor_scores = {}
    for donor in pd_clients:
        if donor == target_cid or donor_ban_counter.get(donor, 0) > 0 or not perf_hist.get(donor):
            continue
        donor_role = roles[donor]
        if target_role == "strong" and donor_role != "strong":
            continue
        if target_role == "weak" and donor_role == "weak":
            continue
        cos_val = cosine_delta(client_sds[donor], target_sd, ref_sd, device)
        scale = donor_scale_from_cos(cos_val, cfg)
        last = perf_hist[donor][-1]
        safety = compute_client_safety_gate(donor, perf_hist, cfg)
        base = compute_client_weight(last["val_loss"], last["val_f1"], dataset_sizes.get(donor, 1), safety, cfg)
        if target_role == "weak" and donor_role == "strong":
            base *= 1.35
        elif target_role == "mid" and donor_role == "strong":
            base *= 1.15
        elif target_role == "strong" and donor_role == "strong":
            base *= 0.90
        donor_scores[donor] = float(max(base * scale, 1e-8))
    if not donor_scores:
        return snapshot_state_dict(target_sd, device=device), {}
    donors = list(donor_scores.keys())
    ws = np.array([donor_scores[d] for d in donors], dtype=np.float64)
    ws = np.maximum(ws, 1e-8)
    ws = ws / ws.sum()
    agg_sd = weighted_average_state_dicts([client_sds[d] for d in donors], ws, device=device)
    return agg_sd, {d: float(w) for d, w in zip(donors, ws.tolist())}


def robust_pd_objective(metrics: Dict[str, Dict[str, float]], pd_clients: List[str]) -> Tuple[float, float]:
    f1s = [float(metrics[c]["val_f1"]) for c in pd_clients]
    losses = [float(metrics[c]["val_loss"]) for c in pd_clients]
    return (float(np.min(f1s)) if f1s else 0.0, float(np.mean(losses)) if losses else float("inf"))


@torch.no_grad()
def eval_current_pd_val(models_pd: Dict[str, PDClientModel], pd_pack: Dict[str, Dict], pd_clients: List[str], device) -> Dict[str, Dict[str, float]]:
    out = {}
    for cid in pd_clients:
        vloss, vacc, vf1 = eval_pd_dl(models_pd[cid], pd_pack[cid]["dl_va"], device)
        out[cid] = {"val_loss": float(vloss), "val_acc": float(vacc), "val_f1": float(vf1)}
    return out


def _best_by_val_loss(hist: List[Dict]) -> Dict:
    if not hist:
        return {}
    idx = int(np.argmin([float(h.get("val_loss", 1e9)) for h in hist]))
    return hist[idx]


def write_run_summary_csv(outdir: str, client_ids: List[str], pd_clients: List[str], seed: int, rounds: int,
                          perf_hist: Dict[str, List[Dict]], personalized_results: Dict[str, Dict]):
    rows = []
    subset_str = "-".join(pd_clients)
    for cid in client_ids:
        best = _best_by_val_loss(perf_hist.get(cid, []))
        pers = personalized_results.get(cid, {})
        rows.append({
            "method": "fed_proposed",
            "run_dir": os.path.basename(outdir),
            "seed": int(seed),
            "rounds": int(rounds),
            "pd_clients": subset_str,
            "client": cid,
            "pure_best_round": int(best.get("round", -1)) if best else -1,
            "pure_best_val_loss": float(best.get("val_loss", np.nan)) if best else np.nan,
            "pure_best_val_acc": float(best.get("val_acc", np.nan)) if best else np.nan,
            "pure_best_val_bal_acc": float(best.get("val_bal_acc", np.nan)) if best else np.nan,
            "pure_best_val_f1": float(best.get("val_f1", np.nan)) if best else np.nan,
            "pure_best_val_weighted_f1": float(best.get("val_weighted_f1", np.nan)) if best else np.nan,
            "pure_best_val_precision_macro": float(best.get("val_precision_macro", np.nan)) if best else np.nan,
            "pure_best_val_recall_macro": float(best.get("val_recall_macro", np.nan)) if best else np.nan,
            "pure_best_val_sensitivity": float(best.get("val_sensitivity", np.nan)) if best else np.nan,
            "pure_best_val_specificity": float(best.get("val_specificity", np.nan)) if best else np.nan,
            "pure_best_val_mcc": float(best.get("val_mcc", np.nan)) if best else np.nan,
            "pure_best_test_acc": float(best.get("test_acc", np.nan)) if best else np.nan,
            "pure_best_test_bal_acc": float(best.get("test_bal_acc", np.nan)) if best else np.nan,
            "pure_best_test_f1": float(best.get("test_f1", np.nan)) if best else np.nan,
            "pure_best_test_weighted_f1": float(best.get("test_weighted_f1", np.nan)) if best else np.nan,
            "pure_best_test_precision_macro": float(best.get("test_precision_macro", np.nan)) if best else np.nan,
            "pure_best_test_recall_macro": float(best.get("test_recall_macro", np.nan)) if best else np.nan,
            "pure_best_test_sensitivity": float(best.get("test_sensitivity", np.nan)) if best else np.nan,
            "pure_best_test_specificity": float(best.get("test_specificity", np.nan)) if best else np.nan,
            "pure_best_test_mcc": float(best.get("test_mcc", np.nan)) if best else np.nan,
            "pers_val_acc": float(pers.get("val_acc", np.nan)),
            "pers_val_bal_acc": float(pers.get("val_bal_acc", np.nan)),
            "pers_val_f1": float(pers.get("val_f1", np.nan)),
            "pers_val_weighted_f1": float(pers.get("val_weighted_f1", np.nan)),
            "pers_val_precision_macro": float(pers.get("val_precision_macro", np.nan)),
            "pers_val_recall_macro": float(pers.get("val_recall_macro", np.nan)),
            "pers_val_sensitivity": float(pers.get("val_sensitivity", np.nan)),
            "pers_val_specificity": float(pers.get("val_specificity", np.nan)),
            "pers_val_mcc": float(pers.get("val_mcc", np.nan)),
            "pers_test_acc": float(pers.get("test_acc", np.nan)),
            "pers_test_bal_acc": float(pers.get("test_bal_acc", np.nan)),
            "pers_test_f1": float(pers.get("test_f1", np.nan)),
            "pers_test_weighted_f1": float(pers.get("test_weighted_f1", np.nan)),
            "pers_test_precision_macro": float(pers.get("test_precision_macro", np.nan)),
            "pers_test_recall_macro": float(pers.get("test_recall_macro", np.nan)),
            "pers_test_sensitivity": float(pers.get("test_sensitivity", np.nan)),
            "pers_test_specificity": float(pers.get("test_specificity", np.nan)),
            "pers_test_mcc": float(pers.get("test_mcc", np.nan)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "run_summary.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_clients", nargs="+", required=True, choices=ALL_PD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=cfg.rounds)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "per_client_logs"), exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(int(args.seed))
    device = torch.device(cfg.device)

    pd_clients = [c.strip().upper() for c in args.pd_clients]
    client_ids = pd_clients[:]
    pd_pack, dataset_sizes = {}, {}

    for cid in pd_clients:
        train_root_name, test_root_name, train_mel_name, test_mel_name = PD_PATHS[cid]
        train_root, test_root = getattr(cfg, train_root_name), getattr(cfg, test_root_name)
        train_mel, test_mel = getattr(cfg, train_mel_name), getattr(cfg, test_mel_name)
        X_all, y_all = PD_SCAN[cid](train_root)
        X_te, y_te = PD_SCAN[cid](test_root)
        _ensure_nonempty(f"PD-{cid} train wav list", len(X_all), f"Check train_root={train_root}")
        _ensure_nonempty(f"PD-{cid} test wav list", len(X_te), f"Check test_root={test_root}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx = np.arange(len(X_all))
        tr_idx, va_idx = next(sss.split(idx, y_all))
        X_tr, X_va = [X_all[i] for i in tr_idx], [X_all[i] for i in va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        ds_tr = PDMelDataset(X_tr, y_tr, train_mel, cfg.ts_max_len_mel, train=True)
        ds_va = PDMelDataset(X_va, y_va, train_mel, cfg.ts_max_len_mel, train=False)
        pd_pack[cid] = {
            "tr_paths": X_tr, "va_paths": X_va, "te_paths": X_te,
            "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
            "dl_tr": DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=pd_collate, num_workers=0),
            "dl_va": DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, collate_fn=pd_collate, num_workers=0),
            "mel_tr": train_mel, "mel_te": test_mel,
        }
        dataset_sizes[cid] = int(len(X_tr))

    first_pd = pd_clients[0]
    sample_base = os.path.splitext(os.path.basename(pd_pack[first_pd]["tr_paths"][0]))[0]
    in_dim_mel = int(_load_ts(pd_pack[first_pd]["mel_tr"], sample_base).shape[1])

    epochs_local = {cid: int(getattr(cfg, PD_EPOCHS[cid])) for cid in pd_clients}
    class_w = {cid: make_class_weights(pd_pack[cid]["y_tr"]) for cid in pd_clients}

    global_shared = SharedMelEncoder(in_dim_mel).to(device)

    def make_pd_model_from(shared: SharedMelEncoder) -> PDClientModel:
        enc = SharedMelEncoder(in_dim_mel)
        enc.load_state_dict(shared.state_dict())
        return PDClientModel(enc)

    models = {cid: make_pd_model_from(global_shared) for cid in pd_clients}
    perf_hist = {cid: [] for cid in client_ids}
    logs = {cid: [] for cid in client_ids}
    prev_w_global = None
    donor_ban_counter = {cid: 0 for cid in pd_clients}
    best_global_sd = None
    best_min_pd_val_f1 = -1.0
    best_mean_pd_val_loss = float("inf")
    best_client_val_loss = {cid: float("inf") for cid in pd_clients}
    best_client_shared_sd = {}
    client_anchor_sd = {cid: snapshot_state_dict(global_shared.state_dict(), to_cpu=True) for cid in pd_clients}
    next_alpha_global = {cid: cfg.alpha_max for cid in pd_clients}

    lr_shared_pd = float(cfg.lr_local)
    lr_private = float(cfg.lr_local * cfg.lr_private_factor)

    for rd in range(1, int(args.rounds) + 1):
        for cid in pd_clients:
            if donor_ban_counter[cid] > 0:
                donor_ban_counter[cid] -= 1

        g_sd = global_shared.state_dict()
        with torch.no_grad():
            for cid in pd_clients:
                blended_sd = {k: next_alpha_global[cid] * g_sd[k].detach().to(device) + (1.0 - next_alpha_global[cid]) * client_anchor_sd[cid][k].detach().to(device) for k in g_sd.keys()}
                models[cid].shared_mel.load_state_dict(blended_sd, strict=True)

        ref_shared_sd = snapshot_state_dict(global_shared.state_dict(), device=device)
        prev_local_shared_sd = {cid: snapshot_state_dict(models[cid].shared_mel.state_dict(), to_cpu=True) for cid in pd_clients}

        for cid in pd_clients:
            models[cid], train_loss = train_pd_local(models[cid], pd_pack[cid]["dl_tr"], device, epochs_local[cid], lr_shared_pd, lr_private, class_w[cid], cid, perf_hist, True, ref_shared_sd, cfg.mu_prox, True, True, cfg.base_smoothing, cfg.max_grad_norm)
            val_m = eval_pd_dl_full(models[cid], pd_pack[cid]["dl_va"], device)
            test_m = eval_pd_multicrop_full(models[cid], pd_pack[cid]["te_paths"], pd_pack[cid]["y_te"], pd_pack[cid]["mel_te"], device, cfg.ts_max_len_mel)
            row = {"round": rd, "train_loss": float(train_loss)}
            row.update({f"val_{k}": float(v) for k, v in val_m.items()})
            row.update({f"test_{k}": float(v) for k, v in test_m.items()})
            perf_hist[cid].append(row)
            logs[cid].append(row)
            pd.DataFrame(logs[cid]).to_csv(os.path.join(args.outdir, "per_client_logs", f"{cid}.csv"), index=False)
            if row["val_loss"] < best_client_val_loss[cid] - 1e-12:
                best_client_val_loss[cid] = float(row["val_loss"])
                best_client_shared_sd[cid] = snapshot_state_dict(models[cid].shared_mel.state_dict(), to_cpu=True)
            if hard_safety_violated(cid, perf_hist, cfg):
                donor_ban_counter[cid] = max(donor_ban_counter[cid], cfg.donor_ban_rounds)
                rollback_sd = best_client_shared_sd.get(cid, prev_local_shared_sd[cid])
                models[cid].shared_mel.load_state_dict({k: v.to(device) for k, v in rollback_sd.items()}, strict=True)

        w_global, safety_map, roles = compute_global_weights(pd_clients, perf_hist, dataset_sizes, prev_w_global, donor_ban_counter, cfg)
        prev_w_global = w_global.copy()

        client_sds = {cid: models[cid].shared_mel.state_dict() for cid in pd_clients}
        usable_clients = [cid for cid in pd_clients if donor_ban_counter.get(cid, 0) == 0]
        if not usable_clients:
            usable_clients = pd_clients[:]
        usable_weights = np.array([w_global[pd_clients.index(cid)] for cid in usable_clients], dtype=np.float64)
        usable_weights = np.maximum(usable_weights, 1e-8)
        usable_weights = usable_weights / usable_weights.sum()
        global_sd_new = weighted_average_state_dicts([client_sds[cid] for cid in usable_clients], usable_weights, device=device)
        global_sd_new = ema_state_dict(global_shared.state_dict(), global_sd_new, beta=cfg.global_ema_beta, device=device)
        global_shared.load_state_dict(global_sd_new, strict=True)

        for cid in pd_clients:
            last = perf_hist[cid][-1]
            alpha = compute_alpha_global(float(last["val_f1"]), float(last["val_loss"]), cfg)
            if is_strong_client(float(last["val_f1"]), cfg) and cid in best_client_shared_sd:
                client_anchor_sd[cid] = best_client_shared_sd[cid]
                next_alpha_global[cid] = cfg.alpha_min
            else:
                pers_sd, _ = build_personalized_server_state(cid, pd_clients, client_sds, ref_shared_sd, perf_hist, dataset_sizes, roles, donor_ban_counter, device, cfg)
                client_anchor_sd[cid] = snapshot_state_dict(pers_sd, to_cpu=True)
                next_alpha_global[cid] = alpha

        met_now = eval_current_pd_val(models, pd_pack, pd_clients, device)
        obj_now = robust_pd_objective(met_now, pd_clients)
        if (obj_now[0] > best_min_pd_val_f1 + 1e-12) or (abs(obj_now[0] - best_min_pd_val_f1) <= 1e-12 and obj_now[1] < best_mean_pd_val_loss - 1e-12):
            best_min_pd_val_f1 = float(obj_now[0])
            best_mean_pd_val_loss = float(obj_now[1])
            best_global_sd = snapshot_state_dict(global_shared.state_dict(), to_cpu=True)

        w_str = " ".join([f"{cid}={w_global[i]:.3f}" for i, cid in enumerate(pd_clients)])
        r_str = " ".join([f"{cid}={roles[cid]}" for cid in pd_clients])
        s_str = " ".join([f"{cid}={safety_map[cid]:.2f}" for cid in pd_clients])
        a_str = " ".join([f"{cid}={next_alpha_global[cid]:.2f}" for cid in pd_clients])
        b_str = " ".join([f"{cid}={donor_ban_counter[cid]}" for cid in pd_clients])
        print(f"[Round {rd:03d}] obj(minF1,meanLoss)={obj_now[0]:.3f},{obj_now[1]:.4f} best(minF1,meanLoss)={best_min_pd_val_f1:.3f},{best_mean_pd_val_loss:.4f} | weights: {w_str} | safety: {s_str} | roles: {r_str} | alpha_global: {a_str} | donor_ban: {b_str}")

    print("\n=== FINAL TEST (pure federated, model selected by validation loss per-client) ===")
    for cid in client_ids:
        best = _best_by_val_loss(perf_hist.get(cid, []))
        if best:
            print(f"{cid}: best_round={best['round']} test_acc={best['test_acc']:.3f} test_f1={best['test_f1']:.3f}")

    print("\n=== PERSONALIZATION STAGE (freeze shared; CE only; L2-SP private) ===")
    personalized_results = {}
    if best_global_sd is None:
        best_global_sd = snapshot_state_dict(global_shared.state_dict(), to_cpu=True)

    for cid in pd_clients:
        chosen_sd = best_client_shared_sd.get(cid, best_global_sd)
        models[cid].shared_mel.load_state_dict({k: v.to(device) for k, v in chosen_sd.items()}, strict=True)
        for p in models[cid].shared_mel.parameters():
            p.requires_grad = False
        for p in models[cid].adapter.parameters():
            p.requires_grad = True
        for p in models[cid].head.parameters():
            p.requires_grad = True
        init_priv = capture_private_params(models[cid], shared_prefix="shared_mel")
        models[cid], _ = train_pd_local(models[cid], pd_pack[cid]["dl_tr"], device, int(cfg.personalization_epochs), 0.0, cfg.lr_personal, class_w[cid], cid, perf_hist, False, None, 0.0, False, True, cfg.base_smoothing, cfg.max_grad_norm, init_priv, cfg.lambda_l2sp)
        val_m = eval_pd_dl_full(models[cid], pd_pack[cid]["dl_va"], device)
        test_m = eval_pd_multicrop_full(models[cid], pd_pack[cid]["te_paths"], pd_pack[cid]["y_te"], pd_pack[cid]["mel_te"], device, cfg.ts_max_len_mel)
        personalized_results[cid] = {}
        personalized_results[cid].update({f"val_{k}": float(v) for k, v in val_m.items() if k != "loss"})
        personalized_results[cid].update({f"test_{k}": float(v) for k, v in test_m.items()})
        print(f"{cid} personalized -> val_acc={val_m['acc']:.3f} val_bal_acc={val_m['bal_acc']:.3f} val_f1={val_m['f1']:.3f} test_acc={test_m['acc']:.3f} test_bal_acc={test_m['bal_acc']:.3f} test_f1={test_m['f1']:.3f}")

    write_run_summary_csv(args.outdir, client_ids, pd_clients, int(args.seed), int(args.rounds), perf_hist, personalized_results)


if __name__ == "__main__":
    main()
