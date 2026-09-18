"""Fixed-budget models. No outer test data enters fitting or model selection."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HGB_PARAMS = dict(max_iter=400, learning_rate=0.06, max_depth=6,
                  min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=False)
GATE_CANDIDATES = ["east_km", "north_km", "seafloor_m", "seafloor_slope"]


def tree(seed=0, small=False):
    params = HGB_PARAMS.copy()
    if small:
        params.update(max_iter=120, max_depth=2, min_samples_leaf=40, l2_regularization=10)
    return HistGradientBoostingRegressor(**params, random_state=seed)


class ResidualExperts:
    """Shared HGB + fixed-strength soft mixture of two residual HGBs.

    Cross-fitted residuals use whole-glider holdouts inside outer training data.
    KMeans and all preprocessing fit on outer training features only. If there
    are fewer than three training gliders, retain just the shared model.
    """

    def __init__(self, seed=0, strength=0.25, min_effective=60, expert_capacity="small"):
        self.seed, self.strength, self.min_effective = seed, strength, min_effective
        # "small": deliberately weak residual experts (default).
        # "full":  experts with the same capacity as the shared/baseline learner.
        self.expert_capacity = expert_capacity

    def weights(self, X):
        z = self.gate_transform.transform(X[self.gate_features])
        distances = self.clusters.transform(z) ** 2
        logits = -distances / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def fit(self, X, y, groups):
        groups = np.asarray(groups)
        self.global_model = tree(self.seed).fit(X, y)
        self.experts = []
        self.details = {"strength": self.strength, "inner_folds": 0,
                        "gate_glider_weights": {}, "active_experts": 0,
                        "expert_capacity": self.expert_capacity}
        if len(np.unique(groups)) < 3:
            self.details["fallback"] = "fewer than three training gliders"
            return self
        residual = np.full(len(y), np.nan)
        splitter = GroupKFold(n_splits=min(3, len(np.unique(groups))))
        for tr, va in splitter.split(X, y, groups):
            inner = tree(self.seed).fit(X.iloc[tr], y[tr])
            residual[va] = y[va] - inner.predict(X.iloc[va])
            self.details["inner_folds"] += 1
        if not np.isfinite(residual).all():
            raise ValueError("Incomplete cross-fitted residuals")
        self.gate_features = [f for f in GATE_CANDIDATES if f in X and X[f].notna().any()]
        self.gate_transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        z = self.gate_transform.fit_transform(X[self.gate_features])
        if len(np.unique(z, axis=0)) < 2:
            self.details["fallback"] = "no distinguishable geographic regimes"
            return self
        self.clusters = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(z)
        distances = self.clusters.transform(z) ** 2
        self.temperature = max(float(np.median(distances.min(axis=1))), 0.1)
        weights = self.weights(X)
        self.details["gate_features"] = self.gate_features
        self.details["gate_centres_standardized"] = self.clusters.cluster_centers_.tolist()
        self.details["gate_glider_weights"] = {
            str(g): weights[groups == g].mean(axis=0).tolist() for g in np.unique(groups)}
        self.details["expert_weight_sum"] = weights.sum(axis=0).tolist()
        for k in range(2):
            if weights[:, k].sum() < self.min_effective:
                self.experts.append(None)
            else:
                self.experts.append(tree(self.seed, small=self.expert_capacity == "small")
                                    .fit(X, residual, sample_weight=weights[:, k]))
        self.details["active_experts"] = sum(e is not None for e in self.experts)
        return self

    def predict(self, X):
        result = self.global_model.predict(X)
        if self.experts:
            weights = self.weights(X)
            for k, expert in enumerate(self.experts):
                if expert is not None:
                    result += self.strength * weights[:, k] * expert.predict(X)
        return result


class TabularResNet:
    """Optional small neural baseline, trained from scratch, with fixed epochs."""

    def __init__(self, device="cuda", seed=0, epochs=100):
        self.device, self.seed, self.epochs = device, seed, epochs

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                     keep_empty_features=True), StandardScaler())
        x = torch.as_tensor(self.transform.fit_transform(X), dtype=torch.float32, device=self.device)
        self.mean, self.scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
        target = torch.as_tensor((y - self.mean) / self.scale, dtype=torch.float32, device=self.device)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU(),
                                          nn.Dropout(0.1), nn.Linear(64, 64))

            def forward(self, values):
                return values + self.body(values)

        self.network = nn.Sequential(nn.Linear(x.shape[1], 64), Block(), Block(),
                                     nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1)).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-3, weight_decay=1e-3)
        self.network.train()
        for _ in range(self.epochs):
            for batch in torch.randperm(len(y), device=self.device).split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = ((self.network(x[batch]).flatten() - target[batch]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                optimizer.step()
        self.network.eval()
        self.details = {"epochs": self.epochs, "parameters": sum(p.numel() for p in self.network.parameters()),
                        "pretrained": False, "device": self.device}
        return self

    def predict(self, X):
        import torch
        values = torch.as_tensor(self.transform.transform(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            result = self.network(values).flatten().cpu().numpy()
        return result * self.scale + self.mean


def gbdt(name, seed):
    """Other GBDT implementations at capacity matched to HGB_PARAMS (400/6/0.06)."""
    if name == "catboost_base":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06,
                                 l2_leaf_reg=1.0, random_seed=seed, verbose=0, allow_writing_files=False)
    if name == "xgb_base":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                            reg_lambda=1.0, tree_method="hist", random_state=seed,
                            n_jobs=4, verbosity=0)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                         reg_lambda=1.0, random_state=seed, n_jobs=4, verbose=-1)


def fit_model(name, X, y, groups, *, seed=0, device="cuda", tabpfn_estimators=4,
              moe_strength=0.25, resnet_epochs=100):
    if name in {"catboost_base", "xgb_base", "lgbm_base"}:
        return gbdt(name, seed).fit(X, y)
    if name in {"hgb_base", "hgb_geo", "hgb_coords", "hgb_phys"}:
        return tree(seed).fit(X, y)
    if name in {"moe_geo", "moe_full"}:
        capacity = "full" if name == "moe_full" else "small"
        return ResidualExperts(seed, moe_strength, expert_capacity=capacity).fit(X, y, groups)
    if name in {"tabpfn_geo", "tabpfn_base"}:
        from tabpfn import TabPFNRegressor
        import os
        from pathlib import Path
        from experiments.data import sha256
        model = TabPFNRegressor(device=device, n_estimators=tabpfn_estimators,
                                random_state=seed, memory_saving_mode=True,
                                fit_mode="low_memory", n_jobs=1).fit(X, y)
        cache = os.environ.get("TABPFN_MODEL_CACHE_DIR")
        model.details = {"pretrained": True, "n_estimators": tabpfn_estimators,
                         "device": device, "fit_mode": "low_memory"}
        if cache:
            model.details["checkpoint_sha256"] = {
                path.name: sha256(path) for path in Path(cache).glob("*regressor*.ckpt")}
        return model
    if name == "resnet_geo":
        return TabularResNet(device, seed, resnet_epochs).fit(X, y)
    raise ValueError(name)
