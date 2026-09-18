"""Inner evaluation reconstructs the outer task, including reference calibration."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments.splits import Fold, fold_targets, make_folds


def inner_folds(d, protocol):
    """Indices refer only to d, which is already restricted to outer training."""
    ix = np.arange(len(d))
    empty = np.array([], dtype=int)
    if protocol == 'logo':
        groups = d.glider.to_numpy()
        if len(np.unique(groups)) < 2:
            return
        for i, (tr, held) in enumerate(GroupKFold(n_splits=min(3, len(np.unique(groups)))).split(d, groups=groups)):
            warmup = np.zeros(len(d), dtype=bool)
            for glider in d.iloc[held].glider.unique():
                selected = d.glider.eq(glider)
                cutoff = d.loc[selected, 'hour_utc'].min() + pd.Timedelta(hours=48)
                warmup |= (selected & (d.hour_utc < cutoff)).to_numpy()
            yield Fold(f'inner_gliders_{i}', tr, held[~warmup[held]], held[warmup[held]])
    elif protocol == 'forward':
        times = np.sort(d.hour_utc.unique())
        for q in (.55, .75):
            cutoff = pd.Timestamp(times[min(int(q * len(times)), len(times)-1)])
            train = (d.hour_utc < cutoff-pd.Timedelta(hours=24)).to_numpy()
            test = ((d.hour_utc >= cutoff) & d.glider.isin(d.loc[train, 'glider'])).to_numpy()
            yield Fold(f'inner_forward_{q}', ix[train], ix[test], empty)
    else:
        yield from make_folds(d, 'spatial', buffer_km=2)


def viable_inner(d, protocol, band):
    result = []
    for fold in inner_folds(d, protocol):
        tr, va, yt, yv, refs = fold_targets(d, band, fold, min_reference=12)
        if len(tr) >= 100 and len(va) >= 24:
            result.append((fold, tr, va, yt, yv, refs))
    return result


def choose_alpha(validation):
    """Each item is (truth, global prediction, gated prediction)."""
    if not validation:
        return 0., {}
    losses = {}
    for alpha in (0., .25, .5, 1.):
        losses[alpha] = float(sum(np.sum((y-((1-alpha)*g+alpha*e))**2) for y,g,e in validation)
                              / sum(len(y) for y,_,_ in validation))
    best = min(losses, key=lambda a: (losses[a], a))
    return best, {str(a): loss for a, loss in losses.items()}
