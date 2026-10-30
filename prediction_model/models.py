from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data_file" / "raw_data.xlsx"

ID_COL = "Reference Code"
LF_COL = "simcreep"
HF_COL = "expcreep"
FEATURE_COLS = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "Co",
    "Ti", "Al", "B", "N", "Fe", "NbTa", "Zr", "W",
    "1TTi", "2TTi", "3TTi", "4TTi",
    "1TTe", "2TTe", "3TTe", "4TTe", "FC", "TestT", "TestS",
]

FOLDS = 10
SEED = 30 
HF_RATIOS = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
MIN_HF = 5
N_RESTARTS = 5
MAX_ITER = 50
Z90 = 1.6448536269514722
OUT_PATH = ROOT_DIR / "result" / f"SF_MF_DELTA_seed{SEED}.xlsx"


def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    required = FEATURE_COLS + [LF_COL, HF_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"data: {missing}")

    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    return (
        df,
        df[FEATURE_COLS].to_numpy(float),
        df[LF_COL].to_numpy(float),
        df[HF_COL].to_numpy(float),
    )


def matern52(X1, X2, ell):
    distance = np.sqrt(np.maximum(
        np.sum((X1[:, None] - X2[None, :]) ** 2, axis=2), 0.0
    ))
    scaled = np.sqrt(5.0) * distance / ell
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def solve(K, y, max_tries=10):
    jitter = 0.0
    for _ in range(max_tries):
        try:
            matrix = K + jitter * np.eye(len(K)) if jitter else K
            factor = cho_factor(matrix, lower=True, check_finite=False)
            return factor, cho_solve(factor, y, check_finite=False)
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    raise np.linalg.LinAlgError("Cholesky failed after adding jitter.")


def scores(y_true, y_pred):
    return {
        "R2_test": float(r2_score(y_true, y_pred)),
        "MAE_test": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_test": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def stable_seed(name, mod=100000):
    return int(zlib.adler32(name.encode("utf-8")) % mod)


def fit_gp(X, y, seed, mode):
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y)) or 1.0
    target = (np.asarray(y, float) - y_mean) / y_scale
    n = len(target)
    ell_min = 0.3 if mode == "sf" else 0.05
    bounds = [
        (np.log(1e-6), np.log(1e6)),
        (np.log(ell_min), np.log(1e2)),
        (np.log(1e-6), np.log(1.0)),
    ]

    def nlml(theta):
        amp, ell, noise = np.exp(theta)
        try:
            K = amp * matern52(X, X, ell) + noise * np.eye(n)
            factor, alpha = solve(K, target)
            log_det = 2.0 * np.log(np.diag(factor[0])).sum()
            return float(
                0.5 * target @ alpha + 0.5 * log_det
                + 0.5 * n * np.log(2 * np.pi)
            )
        except np.linalg.LinAlgError:
            return np.inf

    rng = np.random.default_rng(seed)
    if mode == "sf":
        starts = [np.log([1.0, 1.0, 1e-3])]
        starts += [np.array([rng.uniform(*b) for b in bounds])
                   for _ in range(N_RESTARTS - 1)]
    elif mode == "delta":
        starts = [np.log([1.0, 0.2, 1e-2])]
        init_bounds = [(-2, 2), (-1, 1), (-4, -1)]
        for _ in range(N_RESTARTS - 1):
            start = np.array([rng.uniform(*b) for b in init_bounds])
            starts.append(np.clip(
                start,
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            ))
    else:
        raise ValueError(f"Unknown GP mode: {mode}")

    valid = []
    for start in starts:
        kwargs = {"options": {"maxiter": MAX_ITER}} if mode == "sf" else {}
        result = minimize(
            nlml, start, bounds=bounds, method="L-BFGS-B", **kwargs
        )
        if np.isfinite(result.fun) and (mode == "sf" or result.success):
            valid.append(result)

    best = min(valid, key=lambda result: result.fun) if valid else None
    theta = starts[0] if best is None else best.x
    amp, ell, noise = np.exp(theta)
    K = amp * matern52(X, X, ell) + noise * np.eye(n)
    factor, alpha = solve(K, target)
    return {
        "X": X, "amp": amp, "ell": ell, "noise": noise,
        "mean": y_mean, "scale": y_scale, "factor": factor, "alpha": alpha,
    }


def predict_gp(model, X):
    K_star = model["amp"] * matern52(X, model["X"], model["ell"])
    mean = K_star @ model["alpha"]
    solved = cho_solve(model["factor"], K_star.T, check_finite=False)
    variance = model["amp"] + model["noise"] - np.einsum(
        "ij,ij->j", K_star.T, solved
    )
    return (
        model["mean"] + model["scale"] * mean,
        model["scale"] * np.sqrt(np.maximum(variance, 0.0)),
    )


def fit_mf(X_lf, y_lf, X_hf, y_hf, seed, x0=None, sigma_l=1e-4):
    n_lf, n_hf = len(y_lf), len(y_hf)
    target = np.r_[y_lf, y_hf]
    bounds = [(0.1, 5.0), (0.1, 10.0), (0.05, 5.0), (-12, 0)]

    def covariance(theta):
        rho, ell_lf, ell_delta, log_noise_hf = theta
        K_ll = matern52(X_lf, X_lf, ell_lf) + sigma_l * np.eye(n_lf)
        K_lh = rho * matern52(X_lf, X_hf, ell_lf)
        K_hh = (
            rho**2 * matern52(X_hf, X_hf, ell_lf)
            + matern52(X_hf, X_hf, ell_delta)
            + np.exp(log_noise_hf) * np.eye(n_hf)
        )
        return np.block([[K_ll, K_lh], [K_lh.T, K_hh]])

    def nlml(theta):
        try:
            factor, alpha = solve(covariance(theta), target)
            log_det = 2.0 * np.log(np.diag(factor[0])).sum()
            return float(
                0.5 * target @ alpha + 0.5 * log_det
                + 0.5 * len(target) * np.log(2 * np.pi)
            )
        except np.linalg.LinAlgError:
            return np.inf

    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = [1.0, 1.0, 0.5, np.log(1e-3)]
    starts = [np.asarray(x0, float)]
    starts += [np.array([rng.uniform(*b) for b in bounds])
               for _ in range(N_RESTARTS - 1)]
    results = [
        minimize(
            nlml, start, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": MAX_ITER},
        )
        for start in starts
    ]
    valid = [result for result in results if np.isfinite(result.fun)]
    theta = starts[0] if not valid else min(valid, key=lambda x: x.fun).x
    factor, alpha = solve(covariance(theta), target)
    return {
        "X_lf": X_lf, "X_hf": X_hf, "theta": theta,
        "factor": factor, "alpha": alpha, "sigma_l": sigma_l,
    }


def predict_mf(model, X, observed_lf):
    rho, ell_lf, ell_delta, log_noise_hf = model["theta"]
    X_lf, X_hf = model["X_lf"], model["X_hf"]
    n_test = len(X)

    cross_lf = np.hstack([
        matern52(X, X_lf, ell_lf),
        rho * matern52(X, X_hf, ell_lf),
    ])
    cross_hf = np.hstack([
        rho * matern52(X, X_lf, ell_lf),
        rho**2 * matern52(X, X_hf, ell_lf)
        + matern52(X, X_hf, ell_delta),
    ])
    cross = np.vstack([cross_lf, cross_hf])
    mean = cross @ model["alpha"]
    solved = cho_solve(model["factor"], cross.T, check_finite=False)

    K_lf = matern52(X, X, ell_lf)
    K_cross = rho * K_lf
    K_hf = rho**2 * K_lf + matern52(X, X, ell_delta)
    covariance = np.block([[K_lf, K_cross], [K_cross.T, K_hf]])
    covariance -= cross @ solved
    covariance = (covariance + covariance.T) / 2.0

    mean_lf, mean_hf = mean[:n_test], mean[n_test:]
    S_ll = (
        covariance[:n_test, :n_test]
        + model["sigma_l"] * np.eye(n_test)
    )
    S_hl = covariance[n_test:, :n_test]
    S_lh = covariance[:n_test, n_test:]
    S_hh = covariance[n_test:, n_test:]
    _, inverse_action = solve(S_ll, np.eye(n_test))
    gain = S_hl @ inverse_action
    pred = mean_hf + gain @ (observed_lf - mean_lf)
    cond_cov = S_hh - gain @ S_lh
    variance = np.diag((cond_cov + cond_cov.T) / 2.0) + np.exp(log_noise_hf)
    return pred, np.sqrt(np.maximum(variance, 0.0))


def add_result(runs, points, df, test_idx, fold, ratio,
               hf_count, model_name, y_true, pred, std):
    runs.append({
        "fold": fold,
        "HF_visible_ratio": ratio,
        "model": model_name,
        "n_HF_used": hf_count,
        **scores(y_true, pred),
        "mean_std_test": float(np.mean(std)),
    })

    low, high = pred - Z90 * std, pred + Z90 * std
    sample_ids = (
        df.loc[test_idx, ID_COL].astype(str).to_numpy()
        if ID_COL in df.columns else test_idx.astype(str)
    )
    for i, row_idx in enumerate(test_idx):
        points.append({
            "fold": fold,
            "HF_visible_ratio": ratio,
            "model": model_name,
            "row_index": int(row_idx),
            "sample_id": sample_ids[i],
            "y_true": float(y_true[i]),
            "y_pred": float(pred[i]),
            "pred_std": float(std[i]),
            "PI90_low": float(low[i]),
            "PI90_high": float(high[i]),
            "hit90": bool(low[i] <= y_true[i] <= high[i]),
        })


def summarize(runs, points):
    run_df = pd.DataFrame(runs)
    point_df = pd.DataFrame(points)
    groups = ["HF_visible_ratio", "model"]
    metric_cols = ["R2_test", "MAE_test", "RMSE_test", "mean_std_test"]

    summary_rows = []
    for key, group in run_df.groupby(groups):
        row = dict(zip(groups, key))
        row["n_runs"] = len(group)
        row["n_HF_used_mean"] = float(group["n_HF_used"].mean())
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)

    uncertainty_rows = []
    for key, group in point_df.groupby(groups):
        row = dict(zip(groups, key))
        row["n_points"] = len(group)
        row["PICP90"] = float(group["hit90"].mean())
        row["PI90_width_mean"] = float(
            (group["PI90_high"] - group["PI90_low"]).mean()
        )
        row["pred_std_mean"] = float(group["pred_std"].mean())
        uncertainty_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(groups).reset_index(drop=True)
    uncertainty_df = (
        pd.DataFrame(uncertainty_rows).sort_values(groups).reset_index(drop=True)
    )
    return run_df, summary_df, point_df, uncertainty_df


def main():
    df, X, y_lf, y_hf = load_data(DATA_PATH)
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    runs, points = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), 1):
        X_scaled = StandardScaler().fit(X[train_idx]).transform(X)
        lf_scaled = StandardScaler().fit(
            y_lf[train_idx, None]
        ).transform(y_lf[:, None]).ravel()
        lf_mean = float(y_lf[train_idx].mean())
        lf_scale = float(y_lf[train_idx].std()) or 1.0
        train_order = np.random.default_rng(
            SEED + 10007 * fold
        ).permutation(train_idx)
        last_mf_theta = None

        for ratio in HF_RATIOS:
            n_hf = min(len(train_idx), max(MIN_HF, round(ratio * len(train_idx))))
            hf_idx = train_order[:n_hf]
            X_hf, X_test = X_scaled[hf_idx], X_scaled[test_idx]
            y_hf_train, y_test = y_hf[hf_idx], y_hf[test_idx]
            seed = (
                SEED * 1000003
                + fold * 1009
                + int(ratio * 10000) * 37
                + int(1.0 * 10000) * 17
                + 1
            )

            sf = fit_gp(X_hf, y_hf_train, seed + 13579, mode="sf")
            pred, std = predict_gp(sf, X_test)
            add_result(
                runs, points, df, test_idx, fold, ratio, n_hf,
                "SF_GPR", y_test, pred, std,
            )

            keep_idx = np.random.default_rng(seed).permutation(train_idx)
            mf = fit_mf(
                X_scaled[keep_idx],
                (y_lf[keep_idx] - lf_mean) / lf_scale,
                X_hf,
                (y_hf_train - lf_mean) / lf_scale,
                seed + 777777,
                x0=last_mf_theta,
            )
            last_mf_theta = mf["theta"].copy()
            pred, std = predict_mf(
                mf, X_test, (y_lf[test_idx] - lf_mean) / lf_scale
            )
            add_result(
                runs, points, df, test_idx, fold, ratio, n_hf,
                "MF_DEPLOY", y_test,
                lf_mean + lf_scale * pred, lf_scale * std,
            )

            delta = y_hf_train - y_lf[hf_idx]
            delta_inputs = {
                "DELTA1": (X_hf, X_test, "RES_X"),
                "DELTA2": (
                    np.column_stack([X_hf, lf_scaled[hf_idx]]),
                    np.column_stack([X_test, lf_scaled[test_idx]]),
                    "RES_XyL",
                ),
            }
            for name, (X_train_delta, X_test_delta, old_name) in delta_inputs.items():
                model = fit_gp(
                    X_train_delta,
                    delta,
                    seed + stable_seed(old_name, 99991),
                    mode="delta",
                )
                delta_pred, std = predict_gp(model, X_test_delta)
                add_result(
                    runs, points, df, test_idx, fold, ratio, n_hf,
                    name, y_test, y_lf[test_idx] + delta_pred, std,
                )

            print(f"Fold {fold}/{FOLDS} | HF ratio={ratio:.2f} | HF={n_hf}")

    run_df, summary_df, point_df, uncertainty_df = summarize(runs, points)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        run_df.to_excel(writer, sheet_name="PerRun", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        point_df.to_excel(writer, sheet_name="Pointwise", index=False)
        uncertainty_df.to_excel(writer, sheet_name="UncertaintySummary", index=False)
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
