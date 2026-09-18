"""Matched-family summaries; diagnostic raw experts stay explicitly labeled."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.report import paired_day_bootstrap, score


def build_report(root):
    root = Path(root)
    files = sorted((root/'predictions').glob('*.csv'))
    if not files:
        return
    d = pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    d.to_csv(root/'predictions.csv',index=False)
    summary = []
    for (protocol,band,model),s in d.groupby(['protocol','band','model']):
        family = model.split('_')[0]
        baseline = d[(d.protocol==protocol)&(d.band==band)&(d.model==family+'_global')]
        comp = paired_day_bootstrap(s,baseline)
        if 'delta_rmse_vs_hgb_base' in comp:
            comp['delta_rmse_vs_family_global']=comp.pop('delta_rmse_vs_hgb_base')
        folds=[score(g.measured,g.predicted)['rmse_db'] for _,g in s.groupby('fold')]
        summary.append(dict(protocol=protocol,band=band,model=model,**score(s.measured,s.predicted),
                            macro_fold_rmse_db=float(np.mean(folds)),worst_fold_rmse_db=max(folds),
                            n_folds=len(folds),paired_comparison=comp))
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    flat = pd.DataFrame([{k:v for k,v in x.items() if k!='paired_comparison'} for x in summary])
    flat.to_csv(root/'summary.csv',index=False)
    platform=[]
    for (protocol,band,model,glider),g in d.groupby(['protocol','band','model','glider']):
        platform.append(dict(protocol=protocol,band=band,model=model,glider=glider,**score(g.measured,g.predicted)))
    pd.DataFrame(platform).to_csv(root/'per_platform.csv',index=False)
    gates,finetuning=[],[]
    for file in sorted((root/'details').glob('*.json')):
        protocol,fold,band=file.stem.split('__')
        detail=json.loads(file.read_text())
        for family in ('catboost','tabpfn'):
            if family not in detail:continue
            for kind,record in detail[family]['gates'].items():
                gates.append(dict(protocol=protocol,fold=fold,band=band,family=family,gate=kind,
                                   alpha=record['alpha'],active_experts=record['active_experts'],
                                   out_of_support_fraction=record['out_of_support_fraction'],
                                   selected=kind==detail[family]['selected_gate']))
        for mode in ('head','full'):
            if mode not in detail:continue
            record=detail[mode]
            finetuning.append(dict(protocol=protocol,fold=fold,band=band,mode=mode,
                                   selected_steps=record['selected_steps'],
                                   changed=record['parameter_sha256_before']!=record['parameter_sha256_after'],
                                   trainable_parameters=record['trainable_parameters']))
    if gates:pd.DataFrame(gates).to_csv(root/'gate_choices.csv',index=False)
    if finetuning:pd.DataFrame(finetuning).to_csv(root/'finetuning_choices.csv',index=False)
    primary=flat[~flat.model.str.contains('_raw_')]
    headline=primary.pivot_table(index='model',columns='protocol',values='rmse_db',aggfunc='mean')
    headline.to_csv(root/'headline.csv')
    manifest=json.loads((root/'manifest.json').read_text())
    lines=['# Nested expert/adaptation comparison','',f'Run: `{root.name}`. Status: **{manifest["status"]}**.',
           '', 'Observed-input sensitivity study on one mission. Outer folds were inspected in earlier work;',
           'all choices within this campaign use inner training-only validation. Bootstrap intervals are exploratory.',
           '', '## Mean of band-specific pooled RMSEs, dB','',
           '| Model | '+ ' | '.join(headline.columns)+' |',
           '|---|'+ '|'.join(['---:']*len(headline.columns))+'|']
    for name,row in headline.iterrows():
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in row)+' |')
    lines += ['', '## Per-band results','',
              '| Protocol | Band | Model | RMSE dB | MAE dB | Skill vs reference |',
              '|---|---|---|---:|---:|---:|']
    for row in primary.itertuples():
        lines.append(f'| {row.protocol} | {row.band} | {row.model} | {row.rmse_db:.3f} | {row.mae_db:.3f} | {row.skill_vs_zero:.3f} |')
    lines += ['', 'Raw expert predictions are retained in summary.csv as diagnostic arms; they do not choose gates or blend strengths.',
              'See gate_choices.csv or finetuning_choices.csv for fallback/selection counts, and details/ for inner validation evidence.',
              'The physics variables are proxies, not observed mixing, tidal phase, or noise-source labels.',
              'SPWH and UNDO overlap spectrally; five band scores are not five independent replications.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax=headline.plot.bar(figsize=(13,6),ylabel='Mean band RMSE (dB)',rot=40)
    ax.figure.tight_layout();ax.figure.savefig(root/'comparison.png',dpi=160);plt.close(ax.figure)
    print(headline.round(4).to_string(),flush=True)


if __name__=='__main__':
    import sys
    build_report(sys.argv[1])
