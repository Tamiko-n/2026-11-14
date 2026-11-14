"""Training, decision-threshold tuning (validation only) and evaluation."""
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from . import models as M


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    reg_weight: float = 1.0        # weight of the power-regression loss
    class_weighting: bool = True   # pos_weight = #OFF/#ON per appliance
    tune_threshold: bool = True    # pick decision threshold on validation (else 0.5)
    seed: int = 42
    max_train: int = 150_000       # cap on training windows (CPU budget); 0 = all
    focal_gamma: float = 0.0

    def to_dict(self):
        return asdict(self)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    probs, pows = [], []
    for i in range(0, len(X), bs):
        lo, pw = model(_t(X[i:i + bs]))
        probs.append(torch.sigmoid(lo).numpy())
        pows.append(pw.numpy())
    return np.concatenate(probs), np.concatenate(pows)


def tune_thresholds(prob, y):
    """Per-appliance probability cut-off that maximises macro-F1 on VALIDATION."""
    out = []
    for j in range(y.shape[1]):
        valid = np.isfinite(y[:, j])
        truth = y[valid, j] == 1
        scores = prob[valid, j]
        if not truth.any() or truth.all():
            out.append(0.5)
            continue
        best, bt = -1, 0.5
        for t in np.linspace(0.05, 0.95, 37):
            prediction = scores > t
            tp = np.count_nonzero(truth & prediction)
            tn = np.count_nonzero(~truth & ~prediction)
            errors = len(truth) - tp - tn
            f = tp / max(2 * tp + errors, 1) + tn / max(2 * tn + errors, 1)
            if f > best:
                best, bt = f, t
        out.append(float(bt))
    return np.array(out, np.float32)


def train_torch(name, data, cfg: TrainConfig, app_idx=None, log=print, progress=None):
    """Train one network. app_idx restricts to a subset of appliances (single-task logic)."""
    set_seed(cfg.seed)
    A_all = data["appliances"]
    app_idx = list(range(len(A_all))) if app_idx is None else list(app_idx)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    scale = np.array([data["p_scale"][A_all[j]] for j in app_idx], np.float32)
    Xtr, Ytr, Ptr = tr["X"], tr["Y"][:, app_idx], tr["P"][:, app_idx] / scale
    if cfg.max_train and len(Xtr) > cfg.max_train:
        sel = np.random.RandomState(cfg.seed).choice(len(Xtr), cfg.max_train, replace=False)
        Xtr, Ytr, Ptr = Xtr[sel], Ytr[sel], Ptr[sel]
    Xva, Yva, Pva = va["X"], va["Y"][:, app_idx], va["P"][:, app_idx] / scale

    W = Xtr.shape[1]
    model = M.build(name, W, len(app_idx))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    pos = np.nan_to_num(np.nanmean(Ytr, axis=0), nan=0.5).clip(1e-3, 1 - 1e-3)
    pw = torch.tensor((1 - pos) / pos if cfg.class_weighting else np.ones_like(pos), dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw.clamp(max=20), reduction="none")
    huber = nn.SmoothL1Loss(beta=0.05, reduction="none")

    def loss_fn(lo, pr, y, p):
        mask = torch.isfinite(y)
        labels = y.nan_to_num()
        classification = bce(lo, labels)
        if cfg.focal_gamma:
            error_probability = (lo.sigmoid() - labels).abs()
            classification = classification * error_probability.pow(cfg.focal_gamma)
        losses = classification + cfg.reg_weight * huber(pr, p.nan_to_num())
        return (losses * mask).sum() / mask.sum().clamp(min=1)

    Xtr_t, Ytr_t, Ptr_t = _t(Xtr), _t(Ytr), _t(Ptr)
    Xva_t, Yva_t, Pva_t = _t(Xva), _t(Yva), _t(Pva)
    hist, best, best_state, bad = [], np.inf, None, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            if len(b) == 1:
                b = torch.cat((b, perm[:1]))
            opt.zero_grad()
            lo, pr = model(Xtr_t[b])
            loss = loss_fn(lo, pr, Ytr_t[b], Ptr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            vprob = []
            for i in range(0, len(Xva_t), 512):
                lo, pr = model(Xva_t[i:i + 512])
                vl += loss_fn(lo, pr, Yva_t[i:i + 512], Pva_t[i:i + 512]).item() * len(lo)
                n += len(lo)
                vprob.append(torch.sigmoid(lo).numpy())
            vl /= n
        vprob = np.concatenate(vprob)
        valid = np.isfinite(Yva)
        vf1 = float(np.mean([f1_score(Yva[valid[:, j], j], vprob[valid[:, j], j] > 0.5,
                                     average="macro", zero_division=0)
                             for j in range(len(app_idx)) if valid[:, j].any()]))
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_loss": vl, "val_macro_f1": vf1,
                     "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0})
        log(f"[{name}] epoch {ep:02d}  train {tot / nb:.4f}  val {vl:.4f}  val-F1 {vf1:.3f}")
        if progress:
            progress(ep / cfg.epochs)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"[{name}] early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    vprob, _ = predict(model, Xva)
    thr = tune_thresholds(vprob, Yva) if cfg.tune_threshold else np.full(len(app_idx), 0.5, np.float32)
    return model, hist, thr, time.time() - t0


# ---------------------------------------------------------------- metrics
def metrics(y, prob, p_true_w, p_pred_w, thr):
    """Per-appliance classification + regression metrics."""
    yhat = (prob > thr).astype(int)
    y = y.astype(int)
    p_gated = np.where(yhat == 1, np.maximum(p_pred_w, 0), 0.0)   # OFF => 0 W
    err = p_gated - p_true_w
    tp = int(np.count_nonzero((y == 1) & (yhat == 1)))
    tn = int(np.count_nonzero((y == 0) & (yhat == 0)))
    fp = int(yhat.sum()) - tp
    fn = int(y.sum()) - tp
    denom_on, denom_off = tp + fn, tn + fp
    recall = tp / max(denom_on, 1)
    specificity = tn / max(denom_off, 1)
    class_count = int(denom_on > 0) + int(denom_off > 0)
    f1_on = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_off = 2 * tn / max(2 * tn + fp + fn, 1)
    observed_classes = int(tp + fp + fn > 0) + int(tn + fp + fn > 0)
    e_true, e_pred = p_true_w.sum(), p_gated.sum()
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / class_count,
        "macro_f1": (f1_on + f1_off) / observed_classes,
        "precision_on": tp / max(tp + fp, 1),
        "recall_on": recall,
        "mae_w": float(np.abs(err).mean()),
        "rmse_w": float(np.sqrt((err ** 2).mean())),
        "sae": float(abs(e_pred - e_true) / max(e_true, 1e-6)),     # signal aggregate error
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_on": int(y.sum()), "n_off": int(len(y) - y.sum()),
    }


MIN_T = {"accuracy": 0.85, "balanced_accuracy": 0.75, "macro_f1": 0.70}
HIGH_T = {"accuracy": 0.90, "balanced_accuracy": 0.85, "macro_f1": 0.80}


def grade(m):
    if all(m[k] >= v for k, v in HIGH_T.items()):
        return "Excellent"
    if all(m[k] >= v for k, v in MIN_T.items()):
        return "Minimum"
    return "Below"
