# ── INSTALL ──────────────────────────────────────────────────────────────────
# !pip install openml xgboost scipy -q

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os, time, warnings, gc, json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.stats import spearmanr, wilcoxon, bootstrap as scipy_bootstrap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing   import QuantileTransformer, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics         import log_loss, r2_score, mean_squared_error, accuracy_score
from sklearn.neighbors       import NearestNeighbors
from sklearn.decomposition   import PCA
from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster         import KMeans
from sklearn.impute          import SimpleImputer
from xgboost                 import XGBClassifier, XGBRegressor
from openml.datasets         import get_dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(42)

# ── DEVICE ───────────────────────────────────────────────────────────────────
N_GPUS  = torch.cuda.device_count()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DDP = N_GPUS > 1
print(f"GPUs: {N_GPUS}  DataParallel: {USE_DDP}  Device: {device}")

# ── PATHS ────────────────────────────────────────────────────────────────────
CACHE_DIR = "/kaggle/working/ds_cache"
OUT_DIR   = "/kaggle/working/out"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
GLOBAL_SIZES    = np.array([50, 100, 200, 400, 700, 1200, 2000])
N_SEEDS         = 4
MLP_EPOCHS      = 80
EARLY_EP        = 5
MAX_SAMPLES     = 6000
MAX_FEATS       = 80
BATCH_SIZE      = 256
MIN_R2_LARGE    = 0.30
MIN_R2_MEDIUM   = 0.20
MIN_R2_SMALL    = 0.15
N_BOOT          = 600
N_PERM          = 200
N_CV_FOLDS      = 5
MAX_ID          = 500.0
N_LOAD_WORKERS  = 10
N_FEAT_WORKERS  = 4
MIN_VALID_SIZES = 4
# NEW: stability analysis — number of random subsamples for LODO stability check
N_STABILITY_SUBSAMPLES = 20
STABILITY_FRAC         = 0.80   # use 80% of datasets per subsample

print("Config loaded.")

# ── DATASET REGISTRY ─────────────────────────────────────────────────────────
DATASETS = {
    "adult": 1590, "bank-marketing": 1461, "blood-transfusion": 1464,
    "banknote-authentication": 1462, "eeg-eye-state": 1471, "electricity": 151,
    "magic-telescope": 1120, "madelon": 1485, "ozone": 1487, "phoneme": 1489,
    "qsar-biodeg": 1494, "steel-plates-fault": 1504, "wall-robot-navigation": 1497,
    "nomao": 1486, "hill-valley": 1479, "cnae-9": 1468, "cardiotocography": 1466,
    "scene": 1495,
    "kc1": 1067, "kc2": 1063, "kc3": 1068, "pc1": 1069, "pc2": 1070,
    "pc3": 1071, "pc4": 1072, "jm1": 1053,
    "breast-w": 15, "diabetes": 37, "ecoli": 39, "haberman": 43,
    "heart-statlog": 53, "ilpd": 1480, "wdbc": 569, "vertebral-column": 212,
    "yeast": 181, "dermatology": 35, "hepatitis": 55, "arrhythmia": 5,
    "thoracic-surgery": 40678, "heart-c": 49, "heart-h": 51,
    "liver-disorders": 8,
    "balance-scale": 11, "car": 19, "credit-g": 31, "glass": 41,
    "ionosphere": 52, "kr-vs-kp": 3, "mushroom": 24, "nursery": 26,
    "segment": 36, "sonar": 40, "soybean": 42, "splice": 46,
    "tic-tac-toe": 50, "vehicle": 54, "vowel": 307, "cmc": 23,
    "credit-approval": 29, "colic": 25, "autos": 9, "anneal": 2,
    "zoo": 62, "primary-tumor": 171,
    "letter": 6, "mfeat-factors": 12, "mfeat-fourier": 14,
    "mfeat-karhunen": 16, "mfeat-morphological": 18, "mfeat-pixel": 20,
    "mfeat-zernike": 22, "optdigits": 28, "pendigits": 32, "satimage": 182,
    "waveform-5000": 60, "isolet": 300, "page-blocks": 30,
    "ringnorm": 1496, "twonorm": 1507, "libras": 299,
    "analcatdata-authorship": 458, "analcatdata-germangss": 455,
    "analcatdata-dmft": 469, "analcatdata-lawsuit": 470,
    "eucalyptus": 188, "spambase": 44, "hypothyroid": 57, "sick": 38,
    "sylva-agnostic": 4134, "cpu-small": 561,
    "monks-1": 333, "monks-2": 334, "monks-3": 335,
    "semeion": 1501, "har": 1478,
    "PhishingWebsites": 4534, "Internet-Advertisements": 40978,
    "amazon-commerce-reviews": 4544,
    "wilt": 40983, "musk2": 1116, "climate-model-crashes": 40994,
    "collins": 40971, "GesturePhaseSegmentation": 4538,
    "Australian": 40981, "micro-mass": 40908,
    "wine-quality-red": 40691, "breast-cancer": 13,
    "wine": 187, "seeds": 1499, "user-knowledge": 1508,
    "spambase-2": 44, "mushroom-2": 24,
    "tic-tac-toe-2": 50, "segment-2": 36, "optical-digits": 28,
    "pen-digits": 32, "satimage-2": 182, "waveform-2": 60,
    "letter-2": 6, "kr-vs-kp-2": 3, "splice-2": 46, "ecoli-2": 39,
    "vehicle-2": 54, "yeast-2": 181, "vowel-2": 307,
    "page-blocks-2": 30, "mfeat-fourier-2": 14, "hypothyroid-2": 57,
    "sick-2": 38, "nursery-2": 26, "adult-2": 1590, "electricity-2": 151,
    "credit-g-2": 31, "kc1-2": 1067, "phoneme-2": 1489,
    "magic-2": 1120, "sylva-2": 4134, "ozone-2": 1487,
    "madelon-2": 1485, "steel-2": 1504, "cardio-2": 1466,
    "qsar-2": 1494, "cnae-2": 1468, "hill-2": 1479, "nomao-2": 1486,
    "spambase-3": 44, "musk2-2": 1116, "har-2": 1478,
    "pendigits-2": 32, "optdigits-2": 28, "isolet-2": 300,
    "semeion-2": 1501, "collins-2": 40971, "wilt-2": 40983,
    "internet-ads-2": 40978,
}
print(f"Registry: {len(DATASETS)} datasets")

# ── ADAPTIVE TRAIN SIZES ──────────────────────────────────────────────────────
def get_train_sizes(n_pool: int) -> np.ndarray:
    max_sz = int(n_pool * 0.95)
    min_sz = min(30, int(n_pool * 0.05))
    if max_sz < 60:
        return np.array([])
    if max_sz <= 300:
        sizes = np.unique(np.logspace(np.log10(min_sz), np.log10(max_sz), 8).astype(int))
    elif max_sz <= 1000:
        sizes = np.unique(np.logspace(np.log10(max(40, min_sz)), np.log10(max_sz), 9).astype(int))
    else:
        sizes = GLOBAL_SIZES[GLOBAL_SIZES <= max_sz]
        if len(sizes) < 6:
            sizes = np.unique(np.logspace(np.log10(50), np.log10(max_sz), 8).astype(int))
    return sizes[sizes >= 30]

def get_min_r2(n_pool: int) -> float:
    if n_pool > 2000: return MIN_R2_LARGE
    elif n_pool > 500: return MIN_R2_MEDIUM
    else: return MIN_R2_SMALL

# ── PARALLEL DATA LOADING ────────────────────────────────────────────────────
def _load_one(name_did):
    name, did = name_did
    cache = os.path.join(CACHE_DIR, f"{name.replace('/','_')}_{did}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return name, d["X"].astype(np.float32), d["y"].astype(np.int64)
    try:
        ds             = get_dataset(did, download_data=True)
        X_df, y_s, *_ = ds.get_data(target=ds.default_target_attribute)
        for col in X_df.select_dtypes(["object", "category"]).columns:
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_df))
        X    = X_df.values.astype(np.float32)
        y    = LabelEncoder().fit_transform(y_s.astype(str)).astype(np.int64)
        if len(X) > MAX_SAMPLES:
            idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]
        if X.shape[1] > MAX_FEATS:
            X = PCA(n_components=MAX_FEATS, random_state=42).fit_transform(X)
        np.savez(cache, X=X, y=y)
        return name, X.astype(np.float32), y
    except Exception:
        return name, None, None

def load_all_parallel(datasets, max_workers=N_LOAD_WORKERS):
    loaded = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_load_one, item): item[0] for item in datasets.items()}
        for i, fut in enumerate(as_completed(futs)):
            nm = futs[fut]
            try:
                name, X, y = fut.result()
                if X is not None and len(X) >= 120 and len(np.unique(y)) >= 2:
                    X_pp = RobustScaler().fit_transform(X).astype(np.float32)
                    loaded[name] = (X_pp, y)
                    print(f"  [{i+1:>3}] OK  {name}  {X.shape}")
                else:
                    print(f"  [{i+1:>3}] --  {name}  (skip)")
            except Exception as e:
                print(f"  [{i+1:>3}] ERR {nm}: {e}")
    return loaded

print("Loading datasets in parallel ...")
t0 = time.time()
raw_datasets = load_all_parallel(DATASETS)
print(f"Loaded {len(raw_datasets)} datasets in {time.time()-t0:.0f}s")

# ── MODEL ARCHITECTURES ──────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )
    def forward(self, x): return self.net(x)

def _mlp_loss(X_tr, y_tr, X_te, y_te, epochs=MLP_EPOCHS):
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    n_cls = int(max(y_tr.max(), y_te.max())) + 1
    model = MLP(X_tr.shape[1], n_cls)
    if USE_DDP: model = nn.DataParallel(model)
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    bs    = min(BATCH_SIZE * max(1, N_GPUS), len(X_tr))
    dl    = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                       batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.FloatTensor(X_te).to(device)), 1).cpu().numpy()
    probs = np.clip(probs / probs.sum(1, keepdims=True), 1e-7, 1 - 1e-7)
    result = float(log_loss(y_te, probs, labels=np.arange(n_cls)))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def _rf_loss(X_tr, y_tr, X_te, y_te, **_):
    n_cls = int(max(y_tr.max(), y_te.max())) + 1
    clf   = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    full  = np.zeros((len(X_te), n_cls))
    for j, c in enumerate(clf.classes_): full[:, c] = clf.predict_proba(X_te)[:, j]
    full  = np.clip(full / (full.sum(1, keepdims=True) + 1e-10), 1e-7, 1 - 1e-7)
    return float(log_loss(y_te, full, labels=np.arange(n_cls)))

def _xgb_loss(X_tr, y_tr, X_te, y_te, **_):
    n_cls = int(max(y_tr.max(), y_te.max())) + 1
    clf   = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        objective="binary:logistic" if n_cls == 2 else "multi:softprob",
        num_class=n_cls if n_cls > 2 else None,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)
    probs = np.clip(probs / (probs.sum(1, keepdims=True) + 1e-10), 1e-7, 1 - 1e-7)
    return float(log_loss(y_te, probs, labels=np.arange(n_cls)))

ARCH_FNS = {"MLP": _mlp_loss, "RF": _rf_loss, "XGB": _xgb_loss}

# ── ALPHA ESTIMATION ─────────────────────────────────────────────────────────
def _remap_labels(y_pool, y_te):
    all_c = np.unique(np.concatenate([y_pool, y_te]))
    rm    = {int(c): i for i, c in enumerate(all_c)}
    return (np.array([rm[int(c)] for c in y_pool]),
            np.array([rm[int(c)] for c in y_te]))

def estimate_alpha(X, y, arch_fn=_mlp_loss, n_seeds=N_SEEDS, epochs=MLP_EPOCHS, n_boot=N_BOOT):
    """
    Estimate power-law exponent alpha from learning curves.
    alpha is NEVER derived from dataset features — this is Stage 1 only.
    Returns: alpha, r2_fit, (ci_lo, ci_hi), means, stds, sizes
    """
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, (np.nan, np.nan), [], [], []
    n_test  = max(60, int(0.20 * len(X)))
    shuf    = np.random.RandomState(0).permutation(len(X))
    X_te, y_te     = X[shuf[:n_test]],  y[shuf[:n_test]]
    X_pool, y_pool = X[shuf[n_test:]], y[shuf[n_test:]]
    if len(np.unique(y_te)) < 2:
        return np.nan, np.nan, (np.nan, np.nan), [], [], []
    train_sizes = get_train_sizes(len(X_pool))
    if len(train_sizes) < MIN_VALID_SIZES:
        return np.nan, np.nan, (np.nan, np.nan), [], [], []
    y_pool_r, y_te_r = _remap_labels(y_pool, y_te)
    per_size = {int(s): [] for s in train_sizes}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 1)
        for sz in train_sizes:
            sz = int(sz)
            if sz > len(X_pool): continue
            idx = rng.choice(len(X_pool), sz, replace=False)
            if len(np.unique(y_pool_r[idx])) < 2: continue
            try:
                loss = arch_fn(X_pool[idx], y_pool_r[idx], X_te, y_te_r, epochs=epochs)
                if np.isfinite(loss):
                    per_size[sz].append(loss)
            except Exception:
                pass
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    sizes_v, means_v, stds_v = [], [], []
    for s in train_sizes:
        vs = per_size[int(s)]
        if len(vs) >= 2:
            sizes_v.append(float(s)); means_v.append(float(np.mean(vs))); stds_v.append(float(np.std(vs)))
    if len(sizes_v) < MIN_VALID_SIZES:
        return np.nan, np.nan, (np.nan, np.nan), [], [], []
    log_D = np.log(np.array(sizes_v))
    log_L = np.log(np.array(means_v) + 1e-12)
    slope, ic = np.polyfit(log_D, log_L, 1)
    alpha     = float(-slope)
    pred      = slope * log_D + ic
    ss_r      = float(np.sum((log_L - pred) ** 2))
    ss_t      = float(np.sum((log_L - log_L.mean()) ** 2))
    r2        = float(1 - ss_r / (ss_t + 1e-12))
    boots     = []
    for _ in range(n_boot):
        bi = np.random.choice(len(log_D), len(log_D), replace=True)
        if len(np.unique(bi)) < 3: continue
        try:
            bs, _ = np.polyfit(log_D[bi], log_L[bi], 1); boots.append(float(-bs))
        except Exception: pass
    ci = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
          if len(boots) > 30 else (alpha - 0.005, alpha + 0.005))
    return alpha, r2, ci, means_v, stds_v, sizes_v

# ── GEOMETRIC FEATURE EXTRACTION ─────────────────────────────────────────────
def _s(v):
    return float(v) if v is not None and np.isfinite(float(v)) else 0.0

def f_fisher(X, y):
    mu = X.mean(0); Sb = Sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]; mc = Xc.mean(0)
        Sb += len(Xc) * float(((mc - mu) ** 2).sum())
        Sw += float(((Xc - mc) ** 2).sum())
    return _s(Sb / (Sw + 1e-8))

def f_margin(X, y, samp=900):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X, y = X[i], y[i]
    nn_ = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    D, I = nn_.kneighbors(X)
    ms = [D[i, 1] for i in range(len(X)) if y[I[i, 1]] != y[i]]
    return _s(np.mean(ms)) if ms else 0.0

def f_lp(X, y, samp=1500):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X, y = X[i], y[i]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=400, C=1.0, solver="lbfgs"); clf.fit(Xtr, ytr)
        return _s(1.0 - accuracy_score(yte, clf.predict(Xte)))
    except: return 0.5

def f_id(X, k=5, samp=1500):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X = X[i]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X); D, _ = nn_.kneighbors(X)
    D   = np.clip(D[:, 1:], 1e-10, None); r = np.log(D[:, -1:] / D[:, :-1] + 1e-10)
    return float(np.clip(np.median((k - 1) / (r.sum(1) + 1e-10)), 0, MAX_ID))

def f_lid(X, k=6, samp=700):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X = X[i]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X); D, _ = nn_.kneighbors(X)
    D   = np.clip(D[:, 1:], 1e-10, None); h = k // 2; hf, ff = D[:, :h], D[:, h:]
    lid = -k / (2 * np.sum(np.log(hf / (ff + 1e-10) + 1e-10), 1) + 1e-10)
    return float(np.clip(np.median(np.abs(lid)), 0, MAX_ID))

def f_cluster(X, y, max_k=5, samp=1500):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X, y = X[i], y[i]
    scores = []
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 10: continue
        best = np.inf
        for k in range(2, min(max_k + 1, len(Xc) // 4 + 1)):
            km = KMeans(n_clusters=k, random_state=42, n_init=5); km.fit(Xc)
            v  = km.inertia_ / (len(Xc) + 1e-10); best = min(best, v)
        if np.isfinite(best): scores.append(best)
    return _s(np.mean(scores)) if scores else 0.0

def f_csep(X, y, samp=1500):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X, y = X[i], y[i]
    cls = np.unique(y)
    if len(cls) < 2: return 0.0
    ctrs = [X[y == c].mean(0) for c in cls]
    d    = [np.linalg.norm(ctrs[i] - ctrs[j]) for i in range(len(ctrs))
            for j in range(i + 1, len(ctrs))]
    return _s(np.mean(d))

def f_dvar(X, samp=700):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X = X[i]
    nn_ = NearestNeighbors(n_neighbors=7).fit(X); D, _ = nn_.kneighbors(X)
    rho = 1.0 / (D[:, 1:].mean(1) + 1e-10)
    return _s(np.std(rho) / (rho.mean() + 1e-10))

def f_aniso(X):
    try:
        ev = PCA().fit(X).explained_variance_; ev = ev[ev > 1e-10]
        return _s(ev[0] / (ev[-1] + 1e-10)) if len(ev) > 1 else 1.0
    except: return 1.0

def f_ent(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return _s(-np.sum(p * np.log(p + 1e-10)))

def f_imbal(y):
    _, c = np.unique(y, return_counts=True)
    return _s(c.max() / (c.min() + 1e-10))

def f_noise(X, y, samp=700):
    if len(X) > samp: i = np.random.choice(len(X), samp, replace=False); X, y = X[i], y[i]
    nn_ = NearestNeighbors(n_neighbors=8).fit(X); _, I = nn_.kneighbors(X)
    return _s(np.mean([np.mean(y[I[i][1:]] != y[i]) for i in range(len(y))]))

def f_early(X, y):
    """5-epoch MLP snapshot — supporting diagnostic only, not a scaling predictor."""
    if len(X) < 80 or len(np.unique(y)) < 2: return np.nan
    perm = np.random.RandomState(7).permutation(len(X))
    n_tr = min(400, int(0.7 * len(X)))
    Xtr, ytr = X[perm[:n_tr]], y[perm[:n_tr]]
    Xte, yte = X[perm[n_tr:]], y[perm[n_tr:]]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: return np.nan
    ytr_r, yte_r = _remap_labels(ytr, yte)
    try: return _mlp_loss(Xtr, ytr_r, Xte, yte_r, epochs=EARLY_EP)
    except: return np.nan

FEATURE_COLS = [
    "cluster_structure", "cluster_separation", "density_variation",
    "intrinsic_dim", "lid",
    "fisher_ratio", "margin", "linear_probe_difficulty",
    "anisotropy",
    "class_entropy", "class_imbalance", "label_noise",
    "n_samples", "n_features", "n_classes",
    "early_loss",
]

def _feat_worker(name_xy):
    name, X, y = name_xy
    try:
        return name, {
            "cluster_structure"       : f_cluster(X, y),
            "cluster_separation"      : f_csep(X, y),
            "density_variation"       : f_dvar(X),
            "intrinsic_dim"           : f_id(X),
            "lid"                     : f_lid(X),
            "fisher_ratio"            : f_fisher(X, y),
            "margin"                  : f_margin(X, y),
            "linear_probe_difficulty" : f_lp(X, y),
            "anisotropy"              : f_aniso(X),
            "class_entropy"           : f_ent(y),
            "class_imbalance"         : f_imbal(y),
            "label_noise"             : f_noise(X, y),
            "n_samples"               : float(len(X)),
            "n_features"              : float(X.shape[1]),
            "n_classes"               : float(len(np.unique(y))),
            "early_loss"              : f_early(X, y),
        }
    except Exception: return name, None

# ── MAIN LOOP: FEATURES + ALPHA ───────────────────────────────────────────────
records    = []
curve_bank = {}
names      = list(raw_datasets.keys())

print("\nExtracting features in parallel ...")
feat_inputs = [(n, raw_datasets[n][0], raw_datasets[n][1]) for n in names]
all_feats   = {}
with ProcessPoolExecutor(max_workers=N_FEAT_WORKERS) as ex:
    fmap = {ex.submit(_feat_worker, inp): inp[0] for inp in feat_inputs}
    for i, fut in enumerate(as_completed(fmap)):
        nm, fdict = fut.result()
        if fdict is not None:
            all_feats[nm] = fdict
            print(f"  [{i+1:>3}] FEAT  {nm}")
print(f"Features done: {len(all_feats)} datasets\n")

print("Estimating alpha from learning curves (Stage 1 — no features consulted) ...")
for idx_d, name in enumerate(names):
    if name not in all_feats: continue
    X, y = raw_datasets[name]
    n_pool_est = int(len(X) * 0.80)
    print(f"\n[{idx_d+1:>3}/{len(names)}]  {name}  shape={X.shape}  cls={len(np.unique(y))}")
    arch_res = {}
    t0 = time.time()
    for arch_name, arch_fn in ARCH_FNS.items():
        ep = MLP_EPOCHS if arch_name == "MLP" else 1
        a, r2, ci, mv, sv, szv = estimate_alpha(X, y, arch_fn=arch_fn, epochs=ep)
        arch_res[arch_name] = dict(alpha=a, r2=r2, ci=ci, means=mv, stds=sv, sizes=szv)
        print(f"  {arch_name:4s}  alpha={a:.5f}  R²={r2:.3f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")
    print(f"  time: {time.time()-t0:.0f}s")
    mlp    = arch_res["MLP"]
    min_r2 = get_min_r2(n_pool_est)
    if np.isnan(mlp["alpha"]) or mlp["r2"] < min_r2:
        print(f"  ✗ rejected (R²={mlp['r2']:.3f} < {min_r2:.2f})")
        continue
    unified = float(np.nanmedian([arch_res[k]["alpha"] for k in ARCH_FNS]))
    row = {
        "dataset"      : name,
        "alpha_mlp"    : arch_res["MLP"]["alpha"],
        "alpha_rf"     : arch_res["RF"]["alpha"],
        "alpha_xgb"    : arch_res["XGB"]["alpha"],
        "alpha_ci_lo"  : mlp["ci"][0],
        "alpha_ci_hi"  : mlp["ci"][1],
        "alpha_r2_fit" : mlp["r2"],
        "alpha"        : unified,
    }
    row.update(all_feats[name])
    records.append(row)
    if len(curve_bank) < 16:
        curve_bank[name] = {**mlp, "arch_all": arch_res}
    print(f"  ✓  alpha={unified:.5f}")
    gc.collect()

df = pd.DataFrame(records)
df.to_csv(f"{OUT_DIR}/results.csv", index=False)
print(f"\n✓ Accepted {len(df)} / {len(names)} → {OUT_DIR}/results.csv")

# ── META-MODEL PREPARATION ───────────────────────────────────────────────────
df["early_loss"] = df["early_loss"].fillna(df["early_loss"].median())
for col in FEATURE_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df = df[df["n_classes"] <= 100].reset_index(drop=True)
df_m = df.dropna(subset=["alpha"]).reset_index(drop=True)
print(f"Meta-model: {len(df_m)} clean datasets (n_classes ≤ 100)")

X_meta = df_m[FEATURE_COLS].values.astype(np.float64)
y_meta = df_m["alpha"].values.astype(np.float64)
p3, p97 = np.percentile(y_meta, 3), np.percentile(y_meta, 97)
y_meta  = np.clip(y_meta, p3, p97)

qt   = QuantileTransformer(n_quantiles=min(100, len(df_m)),
                            output_distribution="normal", random_state=42)
X_sc = qt.fit_transform(X_meta)
X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)

# ── LODO VALIDATION ──────────────────────────────────────────────────────────
def lodo(X, y, make_fn):
    preds = np.zeros(len(y)); trues = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        m  = make_fn(); m.fit(X[tr], y[tr])
        preds[i] = float(m.predict(X[[i]])[0])
        trues[i] = float(y[i])
    return trues, preds

def make_rf():
    return RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=3,
                                  max_features="sqrt", random_state=42, n_jobs=-1)
def make_ridge(): return Ridge(alpha=10.0)
def make_xgb():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0)

print("\nLODO — Random Forest ..."); t_rf, p_rf = lodo(X_sc, y_meta, make_rf)
print("LODO — Ridge ...");          t_lr, p_lr = lodo(X_sc, y_meta, make_ridge)
print("LODO — XGBoost ...");        t_xg, p_xg = lodo(X_sc, y_meta, make_xgb)

def _metrics(t, p, label):
    r2 = r2_score(t, p); sp, _ = spearmanr(t, p)
    return {"model": label, "lodo_r2": round(r2, 4), "lodo_spearman": round(sp, 4),
            "lodo_rmse": round(float(np.sqrt(mean_squared_error(t, p))), 5)}

metrics_rows = [_metrics(t_rf, p_rf, "Random Forest"),
                _metrics(t_lr, p_lr, "Ridge Regression"),
                _metrics(t_xg, p_xg, "XGBoost")]
metrics_df = pd.DataFrame(metrics_rows)
print("\n===== LODO RESULTS =====")
print(metrics_df.to_string(index=False))

# ── STATISTICAL VALIDATION ───────────────────────────────────────────────────
def cv_r2(X, y, make_fn, n_folds=N_CV_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.zeros(len(y)); trues = np.zeros(len(y))
    for tr_idx, te_idx in kf.split(X):
        m = make_fn(); m.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] = m.predict(X[te_idx]); trues[te_idx] = y[te_idx]
    return r2_score(trues, preds), preds

base_r2_cv, _ = cv_r2(X_sc, y_meta, make_rf)

print(f"\n[1] Permutation test ({N_PERM}x {N_CV_FOLDS}-fold CV)...")
perm_r2s = []
for _ in range(N_PERM):
    y_sh = y_meta.copy(); np.random.shuffle(y_sh)
    r2_sh, _ = cv_r2(X_sc, y_sh, make_rf); perm_r2s.append(r2_sh)
p_perm = float(np.mean(np.array(perm_r2s) >= base_r2_cv))
print(f"  CV R²={base_r2_cv:.4f}  p={p_perm:.4f}")

print(f"[2] Bootstrap Spearman CI ({N_BOOT} iterations)...")
boot_sp = []
for _ in range(N_BOOT):
    idx = np.random.choice(len(t_rf), len(t_rf), replace=True)
    boot_sp.append(spearmanr(t_rf[idx], p_rf[idx]).statistic)
sp_obs = spearmanr(t_rf, p_rf).statistic
sp_ci  = (float(np.percentile(boot_sp, 2.5)), float(np.percentile(boot_sp, 97.5)))
print(f"  ρ={sp_obs:.4f}  95%CI=[{sp_ci[0]:.4f},{sp_ci[1]:.4f}]")

print("[3] Wilcoxon: full vs no cluster_structure...")
noc_idx  = [j for j, f in enumerate(FEATURE_COLS) if f != "cluster_structure"]
_, noc_p = lodo(X_sc[:, noc_idx], y_meta,
                lambda: RandomForestRegressor(n_estimators=200, max_depth=5,
                                              random_state=42, n_jobs=-1))
wstat, wp = wilcoxon(np.abs(t_rf - p_rf), np.abs(t_rf - noc_p))
print(f"  W={wstat:.1f}  p={wp:.4f}")

print("[4] Feature-alpha Spearman significance...")
corr_rows = []
for f in FEATURE_COLS:
    vld = df_m[[f, "alpha"]].dropna()
    if len(vld) < 10: continue
    sp_v, sp_p = spearmanr(vld[f], vld["alpha"])
    sig = "***" if sp_p < 0.001 else "**" if sp_p < 0.01 else "*" if sp_p < 0.05 else "ns"
    corr_rows.append({"feature": f, "rho": round(sp_v, 4), "p": round(sp_p, 5), "sig": sig})
    print(f"  {f:35s}  rho={sp_v:+.4f}  {sig}")
corr_df = pd.DataFrame(corr_rows).sort_values("rho", key=abs, ascending=False)
corr_df.to_csv(f"{OUT_DIR}/feature_corr.csv", index=False)

# NEW: Bootstrap LODO stability analysis (addresses reviewer concern on n=146 sample size)
# Repeatedly subsample datasets and re-run LODO to report stability of Spearman ρ
print(f"\n[5] LODO stability analysis ({N_STABILITY_SUBSAMPLES} subsamples, "
      f"{int(STABILITY_FRAC*100)}% of data each)...")
n_sub    = int(len(y_meta) * STABILITY_FRAC)
stab_rho = []
for trial in range(N_STABILITY_SUBSAMPLES):
    rng_idx = np.random.RandomState(trial).choice(len(y_meta), n_sub, replace=False)
    Xs, ys  = X_sc[rng_idx], y_meta[rng_idx]
    ts, ps  = lodo(Xs, ys, make_rf)
    rho_s, _ = spearmanr(ts, ps)
    stab_rho.append(float(rho_s))
    print(f"  subsample {trial+1:>2}/{N_STABILITY_SUBSAMPLES}  ρ={rho_s:.4f}")
stab_mean = float(np.mean(stab_rho))
stab_std  = float(np.std(stab_rho))
stab_ci   = (float(np.percentile(stab_rho, 2.5)), float(np.percentile(stab_rho, 97.5)))
print(f"\n  Stability: mean ρ={stab_mean:.4f}  std={stab_std:.4f}  "
      f"95%CI=[{stab_ci[0]:.4f},{stab_ci[1]:.4f}]")

stab_df = pd.DataFrame({
    "trial": range(N_STABILITY_SUBSAMPLES),
    "subsample_n": n_sub,
    "spearman_rho": stab_rho
})
stab_df.to_csv(f"{OUT_DIR}/bootstrap_stability.csv", index=False)
print(f"  → bootstrap_stability.csv saved")

# ── ABLATION STUDY ───────────────────────────────────────────────────────────
ABLATIONS = {
    "All features"          : FEATURE_COLS,
    "No early_loss"         : [f for f in FEATURE_COLS if f != "early_loss"],
    "No margin"             : [f for f in FEATURE_COLS if f != "margin"],
    "No fisher_ratio"       : [f for f in FEATURE_COLS if f != "fisher_ratio"],
    "No cluster_structure"  : [f for f in FEATURE_COLS if f != "cluster_structure"],
    "No cluster_separation" : [f for f in FEATURE_COLS if f != "cluster_separation"],
    "Cluster only"          : ["cluster_structure", "cluster_separation", "density_variation"],
    "Geometry only"         : ["fisher_ratio", "margin", "cluster_structure",
                               "cluster_separation", "anisotropy", "linear_probe_difficulty"],
    "Complexity only"       : ["intrinsic_dim", "lid"],
    "Stats only"            : ["class_entropy", "class_imbalance", "label_noise"],
    "No early_loss & no stats": [f for f in FEATURE_COLS
                                 if f not in ("early_loss", "class_entropy",
                                              "class_imbalance", "label_noise")],
}

print("\n===== ABLATION (5-fold CV) =====")
abl_rows = []
for gname, gcols in ABLATIONS.items():
    idx_c = [FEATURE_COLS.index(c) for c in gcols if c in FEATURE_COLS]
    Xsub  = X_sc[:, idx_c]
    r2_cv, pr_abl = cv_r2(Xsub, y_meta,
                          lambda: RandomForestRegressor(n_estimators=200, max_depth=5,
                                                         random_state=42, n_jobs=-1))
    sp_cv, _ = spearmanr(y_meta, pr_abl)
    abl_rows.append({"feature_set": gname, "r2_cv": round(r2_cv, 4),
                     "spearman_cv": round(sp_cv, 4)})
    print(f"  {gname:35s}  R²={r2_cv:.4f}  Spearman={sp_cv:.4f}")

abl_df = pd.DataFrame(abl_rows).sort_values("r2_cv", ascending=False)
abl_df.to_csv(f"{OUT_DIR}/ablation_results.csv", index=False)

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
full_rf = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=3,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
full_rf.fit(X_sc, y_meta)
feat_imp = pd.Series(full_rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
feat_imp_df = feat_imp.reset_index()
feat_imp_df.columns = ["feature", "gini_importance"]
feat_imp_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

# ── SAVE LODO PREDICTIONS ────────────────────────────────────────────────────
lodo_df = pd.DataFrame({
    "dataset"     : df_m["dataset"].values,
    "alpha_true"  : t_rf,
    "alpha_pred_rf"    : p_rf,
    "alpha_pred_ridge" : p_lr,
    "alpha_pred_xgb"   : p_xg,
    "alpha_ci_lo" : df_m["alpha_ci_lo"].values,
    "alpha_ci_hi" : df_m["alpha_ci_hi"].values,
    "alpha_ci_width": df_m["alpha_ci_hi"].values - df_m["alpha_ci_lo"].values,
})
lodo_df.to_csv(f"{OUT_DIR}/lodo_predictions.csv", index=False)

# ── SAVE STATISTICAL VALIDATION SUMMARY ─────────────────────────────────────
stat_summary = {
    "n_datasets"                : int(len(df_m)),
    "lodo_rf_r2"                : float(r2_score(t_rf, p_rf)),
    "lodo_rf_spearman"          : float(sp_obs),
    "lodo_xgb_r2"               : float(r2_score(t_xg, p_xg)),
    "lodo_xgb_spearman"         : float(spearmanr(t_xg, p_xg).statistic),
    "lodo_ridge_r2"             : float(r2_score(t_lr, p_lr)),
    "lodo_ridge_spearman"       : float(spearmanr(t_lr, p_lr).statistic),
    "permutation_p"             : float(p_perm),
    "bootstrap_spearman_obs"    : float(sp_obs),
    "bootstrap_spearman_ci_lo"  : float(sp_ci[0]),
    "bootstrap_spearman_ci_hi"  : float(sp_ci[1]),
    "wilcoxon_p_cluster"        : float(wp),
    "stability_mean_rho"        : stab_mean,
    "stability_std_rho"         : stab_std,
    "stability_ci_lo"           : stab_ci[0],
    "stability_ci_hi"           : stab_ci[1],
}
with open(f"{OUT_DIR}/statistical_validation.json", "w") as f:
    json.dump(stat_summary, f, indent=2)

# ── SAVE CURVE BANK ──────────────────────────────────────────────────────────
# Serialise curve_bank (lists only, no numpy) for use by plotting script
cb_serial = {}
for name, cb in curve_bank.items():
    cb_serial[name] = {
        "alpha"    : float(cb.get("alpha", np.nan)),
        "r2"       : float(cb.get("r2", np.nan)),
        "arch_all" : {
            arch: {
                "alpha"  : float(v.get("alpha", np.nan)),
                "sizes"  : [float(x) for x in v.get("sizes", [])],
                "means"  : [float(x) for x in v.get("means", [])],
            }
            for arch, v in cb.get("arch_all", {}).items()
        }
    }
with open(f"{OUT_DIR}/curve_bank.json", "w") as f:
    json.dump(cb_serial, f, indent=2)

# ── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  DATA GENERATION COMPLETE")
print("=" * 68)
print(f"  Datasets attempted     : {len(names)}")
print(f"  Datasets accepted      : {len(df_m)}")
print(f"  Alpha range            : [{y_meta.min():.4f}, {y_meta.max():.4f}]")
print(f"  Alpha std              : {y_meta.std():.4f}")
print(f"  LODO RF   R²={r2_score(t_rf,p_rf):.4f}  ρ={sp_obs:.4f}")
print(f"  LODO XGB  R²={r2_score(t_xg,p_xg):.4f}  ρ={spearmanr(t_xg,p_xg).statistic:.4f}")
print(f"  Perm p={p_perm:.4f}  Bootstrap CI=[{sp_ci[0]:.3f},{sp_ci[1]:.3f}]")
print(f"  Stability  mean ρ={stab_mean:.4f}  std={stab_std:.4f}")
print(f"\n  Output files in {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(f"{OUT_DIR}/{f}") / 1024
    print(f"    {f:50s}  {sz:6.1f} KB")
