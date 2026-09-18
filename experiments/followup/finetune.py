"""Supervised v2 adaptation using the installed estimator's own preprocessing."""
import copy
import hashlib

import numpy as np
import torch

from .models import fit_predictor
from .validation import inner_folds


def state_digest(model, group='all'):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        is_head = name.startswith('decoder_dict.')
        if (group == 'head' and not is_head) or (group == 'body' and is_head):
            continue
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def episode_pools(frame, protocol):
    pools = []
    for fold in inner_folds(frame.reset_index(drop=True), protocol):
        if len(fold.train) >= 32 and len(fold.test) >= 8:
            pools.append((fold.train, fold.test))
    return pools


def trajectory(train, y, query, features, protocol, mode, seed, checkpoints, save_path=None):
    """No query labels enter this function. Returns predictions at fixed steps."""
    from tabpfn.preprocessing import fit_preprocessing_one
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    estimator = fit_predictor('tabpfn', train[features], y, seed)
    model = estimator.model_
    before = state_digest(model)
    body_before = state_digest(model, 'body')
    configs = copy.deepcopy(estimator.executor_.ensemble_configs)
    cat_ix = list(estimator.executor_.cat_ix)
    X = train[features].to_numpy(dtype=np.float64)
    y_normalized = (np.asarray(y)-estimator.y_train_mean_)/estimator.y_train_std_
    pools = episode_pools(train, protocol)
    predictions = {0: np.asarray(estimator.predict(query[features]))}
    detail = {'mode': mode, 'seed': seed, 'checkpoint_steps': list(checkpoints),
              'episode_pools': len(pools), 'episode_fallback_random': not bool(pools),
              'losses': [], 'parameter_sha256_before': before}
    max_step = max(checkpoints)
    if not max_step:
        detail.update(parameter_sha256_after=before, trainable_parameters=0, steps=0)
        return predictions, detail
    model.to('cuda').float()
    for name, p in model.named_parameters():
        p.requires_grad_(mode == 'full' or name.startswith('decoder_dict.'))
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    detail['trainable_parameters'] = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
    criterion = estimator.bardist_.to('cuda')
    for step in range(1, max_step+1):
        if pools:
            ctx_pool, qry_pool = pools[(step-1) % len(pools)]
            ctx = rng.choice(ctx_pool, min(192, len(ctx_pool)), replace=False)
            qry = rng.choice(qry_pool, min(32, len(qry_pool)), replace=False)
        else:
            ix = rng.permutation(len(train))
            nquery = min(32, max(8, len(ix)//4))
            qry, ctx = ix[:nquery], ix[nquery:nquery+192]
        assert not set(ctx) & set(qry)
        config = copy.deepcopy(configs[(step-1) % len(configs)])
        config.subsample_ix = None
        config, preprocessor, xc, yc, categorical = fit_preprocessing_one(
            config, X[ctx], y_normalized[ctx], seed+step, cat_ix=cat_ix)
        xq = preprocessor.transform(X[qry]).X
        yq = y_normalized[qry].copy()
        if config.target_transform is not None:
            yq = config.target_transform.transform(yq.reshape(-1, 1)).ravel()
        if not np.isfinite(yq).all() or not np.isfinite(yc).all():
            raise ValueError('Non-finite transformed episode target')
        x_tensor = torch.as_tensor(np.concatenate([xc, xq])[:, None, :],
                                   dtype=torch.float32, device='cuda')
        y_tensor = torch.as_tensor(yc, dtype=torch.float32, device='cuda')
        targets = torch.as_tensor(yq[:, None], dtype=torch.float32, device='cuda')
        model.to('cuda').float().train()
        # Restore the update mask after inference (see compatibility note below).
        for name, p in model.named_parameters():
            p.requires_grad_(mode == 'full' or name.startswith('decoder_dict.'))
        model.reset_save_peak_mem_factor(None)
        optimizer.zero_grad(set_to_none=True)
        logits = model(None, x_tensor, y_tensor, single_eval_pos=len(ctx),
                       only_return_standard_out=True, categorical_inds=categorical)
        loss = criterion(logits / estimator.softmax_temperature, targets).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite supervised loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(grad_norm):
            raise ValueError('Non-finite gradient norm')
        optimizer.step()
        detail['losses'].append({'step': step, 'loss': float(loss.detach()),
                                 'grad_norm': float(grad_norm), 'context': len(ctx), 'query': len(qry)})
        if step in checkpoints:
            model.eval()
            # v2.0.9's memory decorator evaluates p.requires_grad before checking
            # inference mode and dereferences optional None inputs when every
            # attention parameter is frozen. Restore its normal parameter flags;
            # estimator.predict itself runs under torch.inference_mode(), so no
            # gradients or parameter updates occur during this call.
            model.requires_grad_(True)
            predictions[step] = np.asarray(estimator.predict(query[features]))
    detail['parameter_sha256_after'] = state_digest(model)
    detail['body_sha256_before'] = body_before
    detail['body_sha256_after'] = state_digest(model, 'body')
    if mode == 'head' and detail['body_sha256_after'] != body_before:
        raise AssertionError('Decoder-only training modified the backbone')
    detail['steps'] = max_step
    if detail['parameter_sha256_after'] == before:
        raise AssertionError('Optimizer did not change model parameters')
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, save_path)
        detail['checkpoint_file'] = str(save_path)
    del optimizer, model, estimator
    torch.cuda.empty_cache()
    return predictions, detail
