import argparse, json, os, math, zlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

@dataclass
class PipelineConfig:
    results_dir: str = "./results"
    seq_len: int = 4096
    n_per_class: int = 2000
    random_state: int = 42
    block_sizes: Tuple[int, ...] = (8, 16, 32, 64)
    autocorr_lags: Tuple[int, ...] = (1,2,3,4,5,8,16,32,64,128)
    serial_m_values: Tuple[int, ...] = (2,3)
    apen_m_values: Tuple[int, ...] = (2,3)
    cv_folds: int = 5
    n_estimators: int = 400
    max_depth: Optional[int] = None

def bitstring_to_array(s: str, max_len: Optional[int] = None) -> np.ndarray:
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    arr = arr[(arr == 0) | (arr == 1)]
    if max_len is not None:
        arr = arr[:max_len]
    return arr.astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes()

def bits_pm1(bits: np.ndarray) -> np.ndarray:
    return (bits * 2 - 1).astype(np.int8)

def gen_os_urandom(n_bits: int) -> np.ndarray:
    n_bytes = (n_bits + 7) // 8
    b = np.frombuffer(os.urandom(n_bytes), dtype=np.uint8)
    return np.unpackbits(b, bitorder="big")[:n_bits].astype(np.uint8)

def gen_mt(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)

def gen_lcg(n_bits: int, seed: int = 1, a: int = 1664525, c: int = 1013904223, m: int = 2**32) -> np.ndarray:
    x = seed & 0xFFFFFFFF
    out = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        x = (a * x + c) % m
        out[i] = (x >> 31) & 1
    return out

def gen_xorshift32(n_bits: int, seed: int = 2463534242) -> np.ndarray:
    x = np.uint32(seed)
    out = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = (x >> 31) & 1
    return out

def gen_biased(n_bits: int, p_one: float = 0.52, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(n_bits) < p_one).astype(np.uint8)

def generate_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    rows = []
    for _ in range(cfg.n_per_class):
        rows.append(("os_urandom", 1, gen_os_urandom(cfg.seq_len).tobytes()))
    per = cfg.n_per_class // 4
    for _ in range(per):
        rows.append(("mt", 0, gen_mt(cfg.seq_len, rng).tobytes()))
        rows.append(("lcg", 0, gen_lcg(cfg.seq_len, seed=int(rng.integers(1, 1<<31))).tobytes()))
        rows.append(("xorshift", 0, gen_xorshift32(cfg.seq_len, seed=int(rng.integers(1, 1<<31))).tobytes()))
        rows.append(("biased", 0, gen_biased(cfg.seq_len, p_one=0.52, rng=rng).tobytes()))
    df = pd.DataFrame(rows, columns=["generator", "label", "bits_raw"])
    df = df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    return df

def recover_bits(bits_raw: bytes, n_bits: int) -> np.ndarray:
    u = np.frombuffer(bits_raw, dtype=np.uint8)
    return np.unpackbits(u, bitorder="big")[:n_bits].astype(np.uint8)

def monobit_features(bits: np.ndarray) -> Dict[str, float]:
    n = bits.size
    p1 = bits.mean()
    chi = 4.0 * n * (p1 - 0.5) * (p1 - 0.5)
    return {"ones_ratio": float(p1), "monobit_chi": float(chi)}

def runs_features(bits: np.ndarray) -> Dict[str, float]:
    n = bits.size
    if n == 0:
        return {"runs_per_bit": 0.0, "longest_run1": 0.0, "longest_run0": 0.0, "mean_runlen": 0.0, "std_runlen": 0.0}
    changes = np.diff(bits)
    idx = np.where(changes != 0)[0]
    starts = np.r_[0, idx + 1]
    ends = np.r_[idx, n-1]
    lens = ends - starts + 1
    vals = bits[starts]
    longest1 = float(lens[vals == 1].max()) if np.any(vals == 1) else 0.0
    longest0 = float(lens[vals == 0].max()) if np.any(vals == 0) else 0.0
    return {
        "runs_per_bit": float(lens.size) / n,
        "longest_run1": longest1,
        "longest_run0": longest0,
        "mean_runlen": float(lens.mean()),
        "std_runlen": float(lens.std(ddof=0)),
    }

def block_frequency_features(bits: np.ndarray, block_sizes: Tuple[int, ...]) -> Dict[str, float]:
    out = {}
    for m in block_sizes:
        if bits.size < m:
            out[f"block_freq_m{m}_var"] = np.nan
            continue
        k = bits.size // m
        if k == 0:
            out[f"block_freq_m{m}_var"] = np.nan
            continue
        blocks = bits[:k*m].reshape(k, m)
        pi = blocks.mean(axis=1)
        out[f"block_freq_m{m}_var"] = float(np.var(pi))
    return out

def serial_test(bits: np.ndarray, m: int) -> Dict[str, float]:
    n = bits.size
    if n < m + 1:
        return {f"serial_m{m}_chi": np.nan}
    k = 1 << m
    counts = np.zeros(k, dtype=np.int64)
    idx = 0
    for i in range(n):
        idx = ((idx << 1) | int(bits[i])) & (k - 1)
        if i >= m - 1:
            counts[idx] += 1
    exp = (n - m + 1) / k
    chi = float(((counts - exp) ** 2 / exp).sum())
    return {f"serial_m{m}_chi": chi}

def approximate_entropy(bits: np.ndarray, m: int) -> Dict[str, float]:
    n = bits.size
    if n < m + 1:
        return {f"apen_m{m}": np.nan}
    def phi(mm: int) -> float:
        k = 1 << mm
        counts = np.zeros(k, dtype=np.int64)
        idx = 0
        for i in range(n + mm - 1):
            bit = bits[i % n]
            idx = ((idx << 1) | int(bit)) & (k - 1)
            if i >= mm - 1:
                counts[idx] += 1
        p = counts / n
        p = p[p > 0]
        return float(np.sum(np.log(p)) / n)
    ap = phi(m) - phi(m + 1)
    return {f"apen_m{m}": ap}

def cumulative_sums_features(bits: np.ndarray) -> Dict[str, float]:
    x = bits_pm1(bits).astype(np.int32)
    s = np.cumsum(x)
    return {"cusum_max": float(np.max(s)), "cusum_min": float(np.min(s)), "cusum_abs_mean": float(np.mean(np.abs(s)))}

def autocorr_features(bits: np.ndarray, lags: Tuple[int, ...]) -> Dict[str, float]:
    x = bits_pm1(bits).astype(np.float64)
    x = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    n = x.size
    out = {}
    for L in lags:
        if L >= n:
            out[f"ac_lag{L}"] = np.nan
        else:
            out[f"ac_lag{L}"] = float(abs(np.mean(x[:n-L] * x[L:])))
    return out

def byte_level_features(bits: np.ndarray) -> Dict[str, float]:
    b = bits_to_bytes(bits)
    if len(b) == 0:
        return {"byte_entropy": np.nan, "byte_min_entropy": np.nan, "byte_kl_to_uniform": np.nan, "zlib_ratio": np.nan}
    hist = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p[p > 0]
    H = -float(np.sum(nz * np.log2(nz)))
    minH = -float(np.log2(p.max())) if p.max() > 0 else np.nan
    kl = float(np.sum(p * np.log((p + 1e-12) * 256.0)))
    comp = zlib.compress(b, level=9)
    ratio = len(comp) / max(1, len(b))
    return {"byte_entropy": H, "byte_min_entropy": minH, "byte_kl_to_uniform": kl, "zlib_ratio": ratio}

def von_neumann_unbias_len(bits: np.ndarray) -> Dict[str, float]:
    n = bits.size
    if n < 2:
        return {"vn_output_ratio": 0.0, "vn_ones_ratio": 0.5}
    pairs = bits[:n - (n % 2)].reshape(-1, 2)
    mask01 = (pairs[:, 0] == 0) & (pairs[:, 1] == 1)
    mask10 = (pairs[:, 0] == 1) & (pairs[:, 1] == 0)
    out_len = int(mask01.sum() + mask10.sum())
    if out_len == 0:
        return {"vn_output_ratio": 0.0, "vn_ones_ratio": 0.5}
    ones = int(mask01.sum())
    return {"vn_output_ratio": out_len / pairs.shape[0], "vn_ones_ratio": ones / out_len}

def extract_features_from_bits(bits: np.ndarray, cfg: PipelineConfig) -> Dict[str, float]:
    f = {}
    f.update(monobit_features(bits))
    f.update(runs_features(bits))
    f.update(block_frequency_features(bits, cfg.block_sizes))
    for m in cfg.serial_m_values:
        f.update(serial_test(bits, m))
    for m in cfg.apen_m_values:
        f.update(approximate_entropy(bits, m))
    f.update(cumulative_sums_features(bits))
    f.update(autocorr_features(bits, cfg.autocorr_lags))
    f.update(byte_level_features(bits))
    f.update(von_neumann_unbias_len(bits))
    f["length"] = float(bits.size)
    return f

def extract_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        bits = recover_bits(row["bits_raw"], cfg.seq_len)
        feats = extract_features_from_bits(bits, cfg)
        feats["label"] = int(row["label"])
        feats["generator"] = row["generator"]
        rows.append(feats)
    return pd.DataFrame(rows)

def build_model(cfg: PipelineConfig) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", rf)])

def train_and_evaluate(feats: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, object]:
    y = feats["label"].values.astype(int)
    X = feats.drop(columns=["label", "generator"])
    X = X.fillna(X.mean(numeric_only=True))
    feature_names = X.columns.tolist()
    Xv = X.values.astype(np.float64)

    model = build_model(cfg)
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    accs, aucs = [], []
    for tr, te in skf.split(Xv, y):
        model.fit(Xv[tr], y[tr])
        p = model.predict_proba(Xv[te])[:, 1]
        accs.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        try:
            aucs.append(roc_auc_score(y[te], p))
        except:
            aucs.append(float("nan"))
    model.fit(Xv, y)
    return {"model": model, "feature_names": feature_names, "cv_accuracy_mean": float(np.nanmean(accs)), "cv_auc_mean": float(np.nanmean(aucs)), "classes": [0, 1]}

def persist_artifacts(cfg: PipelineConfig, model: Pipeline, feature_names: List[str], metrics: Dict[str, float]) -> Dict[str, str]:
    os.makedirs(cfg.results_dir, exist_ok=True)
    model_path = os.path.join(cfg.results_dir, "model.joblib")
    meta = {"feature_names": feature_names, "config": asdict(cfg), "metrics": metrics}
    joblib.dump({"model": model, "meta": meta}, model_path)
    meta_path = os.path.join(cfg.results_dir, "metrics.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"model_path": model_path, "metrics_path": meta_path}

def features_for_sequence(bitstring: str, cfg: PipelineConfig) -> Tuple[np.ndarray, List[str]]:
    bits = bitstring_to_array(bitstring, max_len=cfg.seq_len)
    if bits.size < cfg.seq_len:
        pad = cfg.seq_len - bits.size
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    feats = extract_features_from_bits(bits, cfg)
    X = pd.DataFrame([feats]).fillna(method="ffill").fillna(method="bfill")
    X = X.fillna(X.mean(numeric_only=True))
    return X.values.astype(np.float64), X.columns.tolist()

def load_model_flexible(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("meta", {})
    return obj, {}

def run_training(cfg: PipelineConfig) -> Dict[str, object]:
    os.makedirs(cfg.results_dir, exist_ok=True)
    ds = generate_dataset(cfg)
    ds_path = os.path.join(cfg.results_dir, "dataset.feather")
    ds.to_feather(ds_path)
    feats = extract_features(ds, cfg)
    feats_path = os.path.join(cfg.results_dir, "features.feather")
    feats.to_feather(feats_path)
    tr = train_and_evaluate(feats, cfg)
    persisted = persist_artifacts(cfg, tr["model"], tr["feature_names"], {"cv_accuracy_mean": tr["cv_accuracy_mean"], "cv_auc_mean": tr["cv_auc_mean"]})
    report = {"dataset_path": ds_path, "features_path": feats_path, "model_path": persisted["model_path"], "metrics_path": persisted["metrics_path"], "cv_accuracy_mean": tr["cv_accuracy_mean"], "cv_auc_mean": tr["cv_auc_mean"]}
    with open(os.path.join(cfg.results_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report

def run_prediction(cfg: PipelineConfig, model_path: str, bitstring_or_file: str, results_name: str = "prediction.json") -> Dict[str, object]:
    if os.path.isfile(bitstring_or_file):
        with open(bitstring_or_file, "r") as f:
            bitstring = "".join(ch for ch in f.read() if ch in "01")
    else:
        bitstring = "".join(ch for ch in bitstring_or_file if ch in "01")
    X, feat_names = features_for_sequence(bitstring, cfg)
    model, meta = load_model_flexible(model_path)
    proba = float(model.predict_proba(X)[0, 1])
    label = "Truly Random" if proba >= 0.5 else "Pseudo-Random"
    out = {"label": label, "prob_truly_random": proba, "threshold": 0.5, "used_features_count": len(feat_names)}
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, results_name), "w") as f:
        json.dump(out, f, indent=2)
    print(label)
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--n_per_class", type=int, default=2000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--mode", type=str, choices=["train", "predict"], required=True)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--input", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = PipelineConfig(results_dir=args.results_dir, seq_len=args.seq_len, n_per_class=args.n_per_class, random_state=args.random_state)
    if args.mode == "train":
        run_training(cfg)
    else:
        if not args.model_path or not args.input:
            raise ValueError("For predict mode, provide --model_path and --input")
        run_prediction(cfg, args.model_path, args.input)

if __name__ == "__main__":
    main()
