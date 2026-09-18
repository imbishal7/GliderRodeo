"""Static-source physical proxies; no acoustic targets enter these features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from experiments.data import BASE_FEATURES

PHYSICS_FEATURES = ['seafloor_slope', 'relative_depth', 'upslope_current_ms',
                    'along_isobath_current_ms', 'topographic_forcing_proxy_ms',
                    'wave_steepness_proxy']
GATES = {
    'geography': ['east_km', 'north_km', 'seafloor_m', 'seafloor_slope'],
    'topography': ['seafloor_m', 'seafloor_slope', 'relative_depth'],
    'forcing': ['wind_ms', 'hs_m', 'wave_steepness_proxy', 'upslope_current_ms'],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_features(frame, bathy):
    d = frame.copy()
    grid = bathy.pivot(index='latitude', columns='longitude', values='z').sort_index().sort_index(axis=1)
    la, lo = grid.index.to_numpy(), grid.columns.to_numpy()
    dy, dx = np.gradient(-grid.to_numpy(), la * 111320,
                         lo * 111320 * np.cos(np.deg2rad(float(np.mean(la)))))
    points = d[['lat', 'lon']].to_numpy()
    gx = RegularGridInterpolator((la, lo), dx, bounds_error=False, fill_value=np.nan)(points)
    gy = RegularGridInterpolator((la, lo), dy, bounds_error=False, fill_value=np.nan)(points)
    slope = np.hypot(gx, gy)
    u, v = d.current_u_ms.to_numpy(), d.current_v_ms.to_numpy()
    flat = slope < 1e-8
    denominator = np.where(flat, 1., slope)
    d['seafloor_slope'] = slope
    d['bathy_grad_east'] = gx
    d['bathy_grad_north'] = gy
    d['upslope_current_ms'] = np.where(flat, 0., -(u * gx + v * gy) / denominator)
    d['along_isobath_current_ms'] = np.where(flat, 0., (-u * gy + v * gx) / denominator)
    d['topographic_forcing_proxy_ms'] = -(u * gx + v * gy)
    d['relative_depth'] = d.depth_m / d.seafloor_m.where(d.seafloor_m > 0)
    d['wave_steepness_proxy'] = (2 * np.pi * d.hs_m.where(d.hs_m >= 0) /
                                 (9.80665 * d.tp_s.where(d.tp_s > 0) ** 2))
    return d.replace([np.inf, -np.inf], np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--bathy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = pd.read_csv(a.source)
    result = physical_features(raw, pd.read_csv(a.bathy))
    assert result.row_id.equals(raw.row_id)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    audit = {'source_sha256': sha256(a.source), 'bathy_sha256': sha256(a.bathy),
             'output_sha256': sha256(a.output), 'rows': len(result),
             'predictor_features': BASE_FEATURES, 'added_physics': PHYSICS_FEATURES,
             'gates': GATES, 'missing_counts': result[PHYSICS_FEATURES].isna().sum().to_dict(),
             'data_kind': 'observational', 'policy': 'observed environmental/state inputs',
             'note': 'Slope is bilinearly sampled from bathymetry gradients; older suite used nearest-cell slope.'}
    a.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
