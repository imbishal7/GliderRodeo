"""Bounded nested comparisons of direct experts and supervised TabPFN updates."""
import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import config
from experiments.data import BASE_FEATURES
from experiments.report import score
from experiments.run import atomic_json, hardware
from experiments.splits import fold_scales, fold_targets, make_folds
from .data import GATES, PHYSICS_FEATURES, sha256
from .models import expert_prediction, fit_predictor
from .validation import choose_alpha, viable_inner


def expert_comparison(train, y, query, protocol, band, seed, event):
    predictions, details = {}, {}
    inners = viable_inner(train, protocol, band)
    inner_records = [{'name': f.name, 'train_ids': train.iloc[t].row_id.tolist(),
                      'validation_ids': train.iloc[v].row_id.tolist(),
                      'calibration_ids': train.iloc[f.calibration].row_id.tolist(), 'references': refs}
                     for f,t,v,_,_,refs in inners]
    for family in ('catboost', 'tabpfn'):
        pairs = {k: [] for k in GATES}
        records = []
        for fold, tr, va, yt, yv, _ in inners:
            event('inner_experts', family=family, fold=fold.name)
            td, vd = train.iloc[tr], train.iloc[va]
            global_model = fit_predictor(family, td[BASE_FEATURES], yt, seed)
            pg = np.asarray(global_model.predict(vd[BASE_FEATURES]))
            del global_model
            for kind in GATES:
                pe, diagnostic = expert_prediction(family, td, yt, vd, BASE_FEATURES, kind, pg, seed)
                pairs[kind].append((yv, pg, pe))
                records.append({'fold': fold.name, 'gate': kind, 'diagnostics': diagnostic})
        settings = {kind: choose_alpha(pairs[kind]) for kind in GATES}
        event('outer_experts', family=family, alphas={k: v[0] for k,v in settings.items()})
        model = fit_predictor(family, train[BASE_FEATURES], y, seed)
        pg = np.asarray(model.predict(query[BASE_FEATURES]))
        del model
        predictions[f'{family}_global'] = pg
        physical = BASE_FEATURES + PHYSICS_FEATURES
        model = fit_predictor(family, train[physical], y, seed)
        predictions[f'{family}_physics'] = np.asarray(model.predict(query[physical]))
        del model
        family_details = {'inner_folds': inner_records, 'inner_diagnostics': records, 'gates': {}}
        best_kind, best_loss = None, float('inf')
        for kind, (alpha, losses) in settings.items():
            pe, diagnostic = expert_prediction(family, train, y, query, BASE_FEATURES, kind, pg, seed)
            predictions[f'{family}_moe_{kind}'] = (1-alpha)*pg + alpha*pe
            predictions[f'{family}_raw_{kind}'] = pe
            diagnostic.update(alpha=alpha, validation_mse=losses)
            family_details['gates'][kind] = diagnostic
            if losses and losses[str(alpha)] < best_loss-1e-12:
                best_kind, best_loss = kind, losses[str(alpha)]
        predictions[f'{family}_moe_selected'] = predictions[f'{family}_moe_{best_kind}'] if best_kind else pg
        family_details['selected_gate'] = best_kind
        details[family] = family_details
        gc.collect()
    return predictions, details


def finetune_comparison(train, y, query, protocol, band, seed, event, checkpoint_dir, steps):
    from .finetune import trajectory
    predictions, details = {}, {}
    inners = viable_inner(train, protocol, band)
    for mode in ('head', 'full'):
        losses = {step: [0., 0] for step in steps}
        inner_records = []
        for fold, tr, va, yt, yv, refs in inners:
            event('inner_finetune', mode=mode, fold=fold.name, steps=max(steps))
            pred, record = trajectory(train.iloc[tr].reset_index(drop=True), yt,
                                      train.iloc[va], BASE_FEATURES, protocol, mode, seed, steps)
            for step, p in pred.items():
                losses[step][0] += float(np.sum((yv-p)**2))
                losses[step][1] += len(yv)
            record.update(fold=fold.name, train_ids=train.iloc[tr].row_id.tolist(),
                          validation_ids=train.iloc[va].row_id.tolist(), references=refs)
            inner_records.append(record)
        mse = {k: s/n for k,(s,n) in losses.items() if n}
        selected = min(mse, key=lambda k:(mse[k],k)) if mse else 0
        event('outer_finetune', mode=mode, selected_steps=selected, validation_mse=mse)
        pred, record = trajectory(train, y, query, BASE_FEATURES, protocol, mode, seed,
                                  sorted(set([0, selected])),
                                  checkpoint_dir/f'{mode}.pt' if selected else None)
        if 'tabpfn_global' in predictions:
            assert np.allclose(predictions['tabpfn_global'], pred[0]), 'Frozen controls differ'
        predictions['tabpfn_global'] = pred[0]
        predictions[f'tabpfn_ft_{mode}'] = pred[selected]
        record.update(selected_steps=selected, validation_mse=mse, inner=inner_records)
        details[mode] = record
        gc.collect()
    return predictions, details


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--table', type=Path, default=config.DATA/'expert_followup.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['experts','finetune'], required=True)
    p.add_argument('--protocols', nargs='+', default=['logo','forward','spatial'])
    p.add_argument('--bands', nargs='+', default=list(config.BANDS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ft-steps', nargs='+', type=int, default=[0,8,32])
    p.add_argument('--target-scaling', choices=['none','platform'], default='none')
    p.add_argument('--max-seconds', type=int, default=7200)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if any(x not in ['logo','forward','spatial'] for x in a.protocols):
        raise ValueError('Invalid protocol')
    if not set(a.bands) <= set(config.BANDS):
        raise ValueError('Invalid bands')
    if 0 not in a.ft_steps or min(a.ft_steps) < 0:
        raise ValueError('Fine-tuning must include its zero-step control')
    a.output.mkdir(parents=True, exist_ok=True)
    lock = open(a.output/'.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = list(Path(__file__).parent.glob('*.py')) + [config.ROOT/'config.py'] + [
        config.ROOT/'experiments'/name for name in ['data.py','splits.py','report.py','run.py']]
    settings = {k: str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    identity = {'settings': settings, 'table_sha256': sha256(a.table),
                'sources': {str(f.relative_to(config.ROOT)):sha256(f) for f in source}}
    audit = json.loads(a.table.with_suffix('.audit.json').read_text())
    assert audit['output_sha256'] == identity['table_sha256']
    checkpoint = Path(os.environ.get('TABPFN_MODEL_CACHE_DIR',
                       Path.home()/'.cache'/'tabpfn'))/'tabpfn-v2-regressor.ckpt'
    checkpoint_hash = sha256(checkpoint)
    mp = a.output/'manifest.json'
    if mp.exists():
        manifest = json.loads(mp.read_text())
        if not a.resume or manifest['identity'] != identity:
            raise ValueError('Existing run requires identical source, inputs, settings and --resume')
    else:
        manifest = {'identity':identity, 'data_audit':audit, 'hardware':hardware(),
                    'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'checkpoint_sha256':checkpoint_hash, 'status':'running'}
        shutil.copy2(a.table, a.output/'input.csv')
        shutil.copy2(a.table.with_suffix('.audit.json'), a.output/'input.audit.json')
        shutil.copy2(Path(__file__).parent/'PLAN.md', a.output/'PLAN.md')
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        (a.output/'packages.txt').write_text('\n'.join(packages)+'\n')
        with tarfile.open(a.output/'source.tar.gz', 'w:gz') as tar:
            for f in source: tar.add(f, arcname=str(f.relative_to(config.ROOT)))
    for name in ['predictions','metrics','splits','details','checkpoints']:
        (a.output/name).mkdir(exist_ok=True)
    atomic_json(mp, manifest)
    log = open(a.output/'events.jsonl', 'a', buffering=1)
    def event(kind, **kw):
        record = {'utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'event':kind, **kw}
        text = json.dumps(record, allow_nan=False)
        log.write(text+'\n'); print(text, flush=True)
    # Preserve the original table's binary predictor values through the derived
    # CSV write/read; one-ULP tie changes can affect TabPFN preprocessing.
    table = pd.read_csv(a.table, float_precision='round_trip')
    table['hour_utc'] = pd.to_datetime(table.hour_utc, utc=True)
    assert table.row_id.is_unique
    start = time.monotonic()
    completed = failures = 0
    bounded = False
    event('start', stage=a.stage, seed=a.seed)
    try:
        with threadpool_limits(limits=4):
            for protocol in a.protocols:
                for fold in make_folds(table, protocol):
                    for band in a.bands:
                        key = f'{protocol}__{fold.name}__{band}'
                        done = a.output/'metrics'/f'{key}.json'
                        if done.exists():
                            event('resume_skip', unit=key); continue
                        tr, te, yt, yv, refs = fold_targets(table, band, fold)
                        if len(tr)<100 or len(te)<24:
                            event('skip', unit=key, train=len(tr), test=len(te)); continue
                        if time.monotonic()-start > a.max_seconds:
                            bounded=True; raise TimeoutError('Runner budget reached')
                        train, query = table.iloc[tr].reset_index(drop=True), table.iloc[te].copy()
                        if a.target_scaling == 'platform':
                            scales = fold_scales(table, band, fold, refs)
                            train_scale = train.glider.map(scales).fillna(1.).to_numpy(dtype=float)
                            query_scale = query.glider.map(scales).fillna(1.).to_numpy(dtype=float)
                            yt = np.asarray(yt, dtype=float)/train_scale
                        else:
                            query_scale = None
                        atomic_json(a.output/'splits'/f'{key}.json', {
                            'train_row_ids':train.row_id.tolist(),'test_row_ids':query.row_id.tolist(),
                            'calibration_row_ids':table.iloc[fold.calibration].row_id.tolist(),'references_db':refs})
                        event('unit_start', unit=key, train=len(tr), test=len(te))
                        unit_start = time.monotonic()
                        import torch
                        torch.set_num_threads(4)
                        torch.cuda.reset_peak_memory_stats()
                        try:
                            if a.stage=='experts':
                                preds, detail = expert_comparison(train,yt,query,protocol,band,a.seed,event)
                            else:
                                preds, detail = finetune_comparison(train,yt,query,protocol,band,a.seed,event,
                                                                  a.output/'checkpoints'/key,sorted(set(a.ft_steps)))
                            rows, metrics = [], []
                            for name, values in preds.items():
                                if query_scale is not None:
                                    values = np.asarray(values, dtype=float)*query_scale
                                assert len(values)==len(yv) and np.isfinite(values).all()
                                saved = query[['row_id','glider','hour_utc','lat','lon']].copy()
                                saved['protocol'],saved['fold'],saved['band'],saved['model']=protocol,fold.name,band,name
                                saved['measured'],saved['predicted']=yv,values
                                saved['reference_db']=saved.glider.map(refs)
                                rows.append(saved)
                                metrics.append(dict(protocol=protocol,fold=fold.name,band=band,model=name,**score(yv,values)))
                            predfile = a.output/'predictions'/f'{key}.csv'
                            temp = predfile.with_suffix('.tmp')
                            pd.concat(rows).to_csv(temp,index=False);temp.replace(predfile)
                            atomic_json(a.output/'details'/f'{key}.json',detail)
                            result={'unit':key,'metrics':metrics,'elapsed_seconds':time.monotonic()-unit_start,
                                    'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
                            atomic_json(done,result)
                            completed+=1
                            event('unit_complete',unit=key,seconds=result['elapsed_seconds'],gpu_mib=result['gpu_peak_allocated_mib'])
                        except Exception as error:
                            failures+=1
                            event('unit_failed',unit=key,error=str(error),traceback=traceback.format_exc())
                            # Surface a systemic implementation failure promptly.
                            if failures>=2: raise
                        finally:
                            gc.collect();torch.cuda.empty_cache()
    except TimeoutError:
        if not bounded: raise
    except BaseException:
        manifest['status']='failed_or_interrupted';atomic_json(mp,manifest);raise
    finally:
        log.close()
    assert sha256(checkpoint)==checkpoint_hash, 'Original pretrained checkpoint was modified'
    manifest.update(status='budget_reached' if bounded else 'completed_with_errors' if failures else 'complete',
                    latest_execution={'completed_units':completed,'failures':failures,'seconds':time.monotonic()-start})
    atomic_json(mp,manifest)
    from .report import build_report
    build_report(a.output)


if __name__=='__main__':
    main()
