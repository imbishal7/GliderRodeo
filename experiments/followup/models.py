"""Matched global and direct-expert families with training-only gates."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .data import GATES


def fit_predictor(family, X, y, seed):
    if family == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=400, depth=6, learning_rate=.06,
                                  l2_leaf_reg=1., random_seed=seed, thread_count=4,
                                  verbose=0, allow_writing_files=False)
    else:
        import torch
        from tabpfn import TabPFNRegressor
        if not torch.cuda.is_available():
            raise RuntimeError('TabPFN requires the requested CUDA device')
        model = TabPFNRegressor(device='cuda', n_estimators=4, random_state=seed,
                                fit_mode='low_memory', memory_saving_mode=True,
                                inference_precision='auto', n_jobs=1)
    return model.fit(X, y)


class Gate:
    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, d):
        self.features = [c for c in GATES[self.kind] if d[c].notna().any()]
        if not self.features:
            raise ValueError('No supported gate features')
        self.transform = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        z = self.transform.fit_transform(d[self.features])
        if len(np.unique(z, axis=0)) < 2:
            raise ValueError('No distinguishable gate regimes')
        self.clusters = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(z)
        dist2 = self.clusters.transform(z) ** 2
        self.temperature = max(float(np.median(dist2.min(axis=1))), .1)
        self.radius = float(np.quantile(np.sqrt(dist2.min(axis=1)), .99))
        self.labels = self.clusters.labels_
        self.details = {'kind': self.kind, 'features': self.features,
                        'temperature': self.temperature, 'radius99': self.radius,
                        'centres_standardized': self.clusters.cluster_centers_.tolist()}
        return self

    def weights(self, d):
        z = self.transform.transform(d[self.features])
        dist2 = self.clusters.transform(z) ** 2
        logits = -dist2/self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        supported = np.sqrt(dist2.min(axis=1)) <= self.radius
        return weights, supported


def expert_prediction(family, train, y, query, features, kind, global_prediction, seed):
    """Direct experts: identical memberships for the two estimator families."""
    gate = Gate(kind, seed).fit(train)
    weights, supported = gate.weights(query)
    components = []
    detail = dict(gate.details, active_experts=0,
                  out_of_support_fraction=float((~supported).mean()), experts=[])
    for k in range(2):
        mask = gate.labels == k
        subset = train.loc[mask]
        info = {'cluster': k, 'n': int(mask.sum()), 'gliders': int(subset.glider.nunique()),
                'hours': int(subset.hour_utc.nunique()),
                'platform_counts': subset.glider.value_counts().to_dict()}
        if mask.sum() < 80 or subset.glider.nunique() < 2 or subset.hour_utc.nunique() < 24:
            info['fallback'] = 'insufficient rows, hours, or platforms'
            components.append(np.asarray(global_prediction))
        else:
            model = fit_predictor(family, subset[features], np.asarray(y)[mask], seed)
            components.append(np.asarray(model.predict(query[features])))
            detail['active_experts'] += 1
            del model
        detail['experts'].append(info)
    mixture = (np.column_stack(components)*weights).sum(axis=1)
    mixture[~supported] = np.asarray(global_prediction)[~supported]
    detail['mean_test_weights'] = weights.mean(axis=0).tolist()
    return mixture, detail
