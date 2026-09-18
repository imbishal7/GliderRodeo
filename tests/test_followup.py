"""Scientific invariants for the nested follow-up, independent of estimator scores."""
import numpy as np
import pandas as pd

from experiments.followup.data import physical_features
from experiments.followup.models import Gate
from experiments.followup.validation import choose_alpha, viable_inner


def mission():
    t=pd.date_range('2026-01-28',periods=240,freq='h',tz='UTC')
    return pd.DataFrame([dict(row_id=i*len(t)+j,glider=f'g{i}',hour_utc=h,
                              FIWH=50+i+np.sin(j/24),east_km=(j%20)-10,
                              north_km=((j//20)%12)-6)
                         for i in range(4) for j,h in enumerate(t)])


def test_inner_group_validation_calibration_isolated():
    d=mission()
    for fold,tr,va,yt,yv,refs in viable_inner(d,'logo','FIWH'):
        assert not set(d.iloc[tr].glider)&set(d.iloc[va].glider)
        assert not set(tr)&set(va)
        assert not set(fold.calibration)&set(va)
        changed=d.copy()
        changed.loc[va,'FIWH']+=1000
        after={f.name:(a,b,c,e,r) for f,a,b,c,e,r in viable_inner(changed,'logo','FIWH')}[fold.name]
        assert refs==after[-1]
        assert np.allclose(yt,after[2])


def test_inner_forward_has_gap_and_past_only_reference():
    d=mission()
    folds=viable_inner(d,'forward','FIWH')
    assert len(folds)==2
    for fold,tr,va,yt,yv,refs in folds:
        assert d.iloc[va].hour_utc.min()-d.iloc[tr].hour_utc.max()>=pd.Timedelta(hours=24)
        for g,ref in refs.items():
            assert ref==d.iloc[tr].loc[lambda x:x.glider.eq(g),'FIWH'].median()


def test_alpha_can_disable_misdirected_experts():
    y=np.array([1.,2.,3.]);global_p=np.array([1.,1.5,2.5]);bad=2*global_p-y
    assert choose_alpha([(y,global_p,bad)])[0]==0
    assert choose_alpha([(y,global_p,y)])[0]==1
    assert choose_alpha([])[0]==0


def test_gate_does_not_refit_on_query_and_rejects_far_queries():
    rng=np.random.default_rng(1)
    d=pd.DataFrame({'east_km':rng.normal(size=200),'north_km':rng.normal(size=200),
                    'seafloor_m':1000+rng.normal(size=200),'seafloor_slope':rng.uniform(size=200)})
    gate=Gate('geography').fit(d)
    centres=gate.clusters.cluster_centers_.copy()
    q=d.iloc[:3].copy();w,_=gate.weights(q)
    far=q.copy();far['east_km']=1e6
    _,supported=gate.weights(far)
    assert not supported.any()
    assert np.allclose(gate.clusters.cluster_centers_,centres)
    assert np.allclose(gate.weights(q)[0],w)
    assert np.allclose(w.sum(axis=1),1)


def test_physics_gradient_sign_units_and_target_independence():
    bathy=pd.DataFrame([dict(latitude=la,longitude=lo,z=-(1000+lo*111320*.1))
                        for la in [-.01,0,.01] for lo in [-.01,0,.01]])
    d=pd.DataFrame(dict(lat=[0.],lon=[0.],current_u_ms=[2.],current_v_ms=[0.],
                        depth_m=[100.],seafloor_m=[1000.],hs_m=[2.],tp_s=[10.],FIWH=[50.]))
    out=physical_features(d,bathy)
    assert np.isclose(out.seafloor_slope.iloc[0],.1)
    assert np.isclose(out.upslope_current_ms.iloc[0],-2.)
    assert np.isclose(out.topographic_forcing_proxy_ms.iloc[0],-.2)
    assert np.isclose(out.relative_depth.iloc[0],.1)
    d.FIWH=1000
    other=physical_features(d,bathy)
    for c in ['upslope_current_ms','topographic_forcing_proxy_ms','wave_steepness_proxy']:
        assert np.allclose(out[c],other[c])


def test_derived_csv_preserves_original_float_values(tmp_path):
    d=pd.DataFrame({'x':[.10000000000000002,1.2345678901234567,123.00000000000003]})
    file=tmp_path/'derived.csv'
    d.to_csv(file,index=False)
    restored=pd.read_csv(file,float_precision='round_trip')
    assert np.array_equal(d.x.to_numpy(),restored.x.to_numpy())
