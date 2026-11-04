"""Reproduce the tutorial without opening Jupyter.

Expensive operations are explicit: --train, --simulate, --evaluate.
The standard notebook loads the prepared model and cached hydraulic signals.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import os
import uuid
import zipfile
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def _network_text_sha256(path):
    # Universal newlines make Windows checkout line endings harmless.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()


def _atomic_npz(path, **arrays):
    """An interrupted write must never replace a usable prepared cache."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,**arrays)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def save_splits(net, splits):
    fields = {f"{part}_{key}": value for part, data in splits.items()
              for key, value in data.items()}
    fields['sensors'] = np.asarray(net.sensors)
    fields['min_source_pipe_length_m'] = float(net.min_source_pipe_length)
    fields['source_pipe_indices'] = np.asarray(net.source_pipe_indices, dtype=int)
    _atomic_npz(ROOT / 'data' / 'source_splits.npz', **fields)


def load_test_set(net, splits=None, path=None):
    """Read cached TSNet events for eligible original pipes without changing the cache.

    Selection uses the prescribed pipe-length threshold only, never prediction
    errors. If supplied, source splits are checked before any event is scored.
    """
    path = ROOT / 'data' / 'tsnet_test.npz' if path is None else Path(path)
    try:
        with np.load(path, allow_pickle=False) as stored:
            data = {key: stored[key] for key in stored.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError('The TSNet cache cannot be read. Restore the complete project archive or regenerate the hydraulic test set.') from error
    if str(data.get('network_text_sha256', '')) != _network_text_sha256(net.inp_path):
        raise ValueError('Cached hydraulic signals belong to a different INP network. Regenerate the TSNet test set.')
    if (np.asarray(data.get('nominal_pipe_wave_speeds_m_s', [])).shape != net.nominal_pipe_wave_speeds.shape
            or not np.allclose(data['nominal_pipe_wave_speeds_m_s'], net.nominal_pipe_wave_speeds, rtol=1e-12, atol=1e-9)):
        raise ValueError('Cached hydraulic signals use a different nominal wave speed. Regenerate the TSNet test set.')
    raw_pipe = np.asarray(data['pipe'])
    pipe = raw_pipe.astype(int)
    if (pipe.ndim != 1 or not len(pipe) or not np.array_equal(raw_pipe, pipe)
            or np.any((pipe < 0) | (pipe >= net.P))):
        raise ValueError('TSNet cache has invalid original pipe indices. Regenerate the TSNet test set.')
    count = len(pipe)
    xi = np.asarray(data['xi'], dtype=float)
    names = np.asarray(data['pipe_names'])
    if (xi.shape != (count,) or not np.isfinite(xi).all()
            or np.any((xi <= 0) | (xi >= 1)) or names.shape != (count,)
            or not np.array_equal(names, np.asarray(net.pipe_names)[pipe])):
        raise ValueError('TSNet cache source names or positions do not match the network. '
                         'Regenerate the TSNet test set.')
    eligible = np.asarray(net.source_pipe_mask, dtype=bool)
    keep = eligible[pipe]
    retained = int(keep.sum())
    if retained == 0:
        raise ValueError('No cached TSNet events meet the source-pipe length threshold. '
                         'Regenerate the TSNet test set for the current configuration.')
    if splits is not None:
        for p, x in zip(pipe[keep], xi[keep]):
            def present(part):
                source = splits[part]
                return bool(np.any((np.asarray(source['pipe']) == p) & np.isclose(
                    np.asarray(source['xi']), x, atol=1e-12, rtol=0.)))
            if not present('test') or present('train') or present('val'):
                raise ValueError('Cached TSNet sources do not match the current held-out split '
                                 'or overlap training/validation sources. Regenerate the TSNet '
                                 'test set after changing source counts or seed.')
    event_keys = ('heads', 'signal', 'arrivals', 'pipe', 'xi', 'pipe_names', 'actual_dt')
    for key in event_keys:
        if key in data:
            array = np.asarray(data[key])
            if array.ndim == 0 or array.shape[0] != count:
                raise ValueError(f'TSNet cache field {key!r} has an inconsistent event count.')
            data[key] = array[keep]
    if 'metadata_json' in data:
        metadata = json.loads(str(np.asarray(data['metadata_json']).item()))
        if not isinstance(metadata, list) or len(metadata) != count:
            raise ValueError('TSNet cache metadata does not match its event count.')
        data['metadata_json'] = json.dumps([item for item, selected in zip(metadata, keep) if selected])
    selection = {
        'min_source_pipe_length_m': float(net.min_source_pipe_length),
        'network_pipes': int(net.P), 'eligible_source_pipes': int(net.source_pipe_count),
        'cached_events': count, 'retained_events': retained, 'excluded_events': count - retained,
        'rule': 'Original pipe length meets the prescribed threshold; no outcome-based selection.',
        'held_out_split_checked': splits is not None,
    }
    data['source_selection_json'] = json.dumps(selection)
    print(f'TSNet sources: retained {retained}/{count} cached events; '
          f'excluded {count-retained} below {net.min_source_pipe_length:g} m.', flush=True)
    return data


def _simulate_one(job):
    from tsnet_tools import simulate_burst
    index, source, sensors, cache, force, inp_path, wave_speed, nominal_speeds = job
    network_sha = _network_text_sha256(inp_path)
    target = Path(cache) / f"event_{index:03d}.npz"
    if target.exists() and not force:
        try:
            with np.load(target, allow_pickle=False) as z:
                if (str(z['pipe_name']) == source['pipe_name'] and float(z['xi']) == source['xi']
                        and str(z.get('network_text_sha256', '')) == network_sha
                        and np.asarray(z.get('nominal_pipe_wave_speeds_m_s', [])).shape == np.asarray(nominal_speeds).shape
                        and np.allclose(z['nominal_pipe_wave_speeds_m_s'], nominal_speeds, rtol=1e-12, atol=1e-9)):
                    return index, str(target), 'cached'
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass  # Explicit simulation requests may rebuild a damaged cache.
    result = simulate_burst(source['pipe_name'], source['xi'], sensors=sensors,
                            inp_path=inp_path, wave_speed=wave_speed,
                            duration=3., dt=.00104045, rise_time=.02, coefficient=.001/np.sqrt(40.))
    # Store compact traces, but preserve arrivals picked on the original MOC grid.
    t = np.arange(0, result['time'][-1], .01)
    def sample(key):
        return np.column_stack([np.interp(t, result['time'], result[key][:, j])
                               for j in range(len(result['node_names']))]).astype(np.float32)
    _atomic_npz(target, time=t, heads=sample('heads'), signal=sample('signal'),
        node_names=result['node_names'], elevations=result['elevations'],
        arrivals=result['arrivals'], pipe_name=source['pipe_name'], xi=source['xi'],
        pipe=source['pipe'], actual_dt=result['actual_dt'],
        metadata_json=json.dumps(result['metadata']), source_json=json.dumps(result['source']),
        network_text_sha256=network_sha, nominal_pipe_wave_speeds_m_s=np.asarray(nominal_speeds))
    return index, str(target), f"{result['metadata']['runtime_s']:.1f}s"


def simulate_test_set(net, splits, events=21, workers=4, force=False):
    """Choose held-out sources before running the localisation model."""
    cache = ROOT / 'data' / 'tsnet_events'
    cache.mkdir(parents=True, exist_ok=True)
    if int(events) < 1 or int(workers) < 1:
        raise ValueError('events and workers must be positive.')
    eligible = np.asarray(net.source_pipe_indices, dtype=int)
    if not len(eligible):
        raise ValueError('No source pipes meet the minimum length threshold.')
    selected = np.unique(np.linspace(0, len(eligible) - 1, min(int(events), len(eligible))).round().astype(int))
    pipe_ids = eligible[selected]
    test = splits['test']
    sources = []
    for p in pipe_ids:
        candidates = np.flatnonzero(test['pipe'] == p)
        if not len(candidates):
            raise ValueError(f'No held-out sources exist for eligible pipe {net.pipe_names[p]}. '
                             'Regenerate the source splits for this configuration.')
        # The first held-out location is fixed before any hydraulic run or inference.
        i = candidates[0]
        sources.append({'pipe': int(p), 'pipe_name': net.pipe_names[p], 'xi': float(test['xi'][i])})
    manifest = {'selection': f'{len(sources)} evenly spaced eligible pipe indices; choose the first held-out position for each pipe',
                'sources': sources, 'sensor_names': list(net.sensors),
                'min_source_pipe_length_m': float(net.min_source_pipe_length),
                'network_pipes': int(net.P), 'eligible_source_pipes': int(net.source_pipe_count),
                'note': 'Prepared before simulation or localisation. No event is selected by its prediction error.'}
    (ROOT / 'data' / 'tsnet_manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = [(i, source, net.sensors, str(cache), force, net.inp_path, net.wave_speed, net.nominal_pipe_wave_speeds) for i, source in enumerate(sources)]
    paths = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_one, job) for job in jobs]
        for future in as_completed(futures):
            index, path, status = future.result()  # A failed event stops the batch.
            paths[index] = path
            print(f"TSNet {sum(p is not None for p in paths)}/{len(paths)}: {sources[index]['pipe_name']} ({status})", flush=True)
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            records.append({key: archive[key] for key in archive.files})
    for record in records[1:]:
        if not np.array_equal(record['node_names'], records[0]['node_names']):
            raise ValueError('Hydraulic event caches have inconsistent junction ordering.')
    size = min(len(r['time']) for r in records)
    package = {key: np.stack([r[key][:size] for r in records]) for key in ['heads', 'signal']}
    package.update(time=records[0]['time'][:size], node_names=records[0]['node_names'],
        elevations=records[0]['elevations'], arrivals=np.stack([r['arrivals'] for r in records]),
        pipe=np.array([s['pipe'] for s in sources]), xi=np.array([s['xi'] for s in sources]),
        pipe_names=np.array([s['pipe_name'] for s in sources]),
        actual_dt=np.array([float(r['actual_dt']) for r in records]),
        metadata_json=json.dumps([json.loads(str(r['metadata_json'])) for r in records]))
    package['network_text_sha256'] = _network_text_sha256(net.inp_path)
    package['nominal_pipe_wave_speeds_m_s'] = np.asarray(net.nominal_pipe_wave_speeds)
    package['source_selection_json'] = json.dumps({
        'min_source_pipe_length_m': float(net.min_source_pipe_length),
        'network_pipes': int(net.P), 'eligible_source_pipes': int(net.source_pipe_count),
        'cached_events': len(sources), 'retained_events': len(sources), 'excluded_events': 0,
        'rule': manifest['selection'], 'held_out_split_checked': True,
    })
    _atomic_npz(ROOT / 'data' / 'tsnet_test.npz', **package)
    return package


def evaluate(net, splits, model, sigma=.010, quantum=.010, *,
             pressure_noise_std_m=.005, extra_jitter_ms=10., seed=20261104,
             output_dir=None):
    """Evaluate the GNN on held-out locations and independently simulated signals.

    The waveform cases re-run the signal-only picker on the stored 100 Hz
    records. Missing detections stay in the pipe-accuracy denominator. A mean
    network-distance error is undefined if any event cannot be localised.
    Nothing is written unless output_dir is supplied.
    """
    from graph_tools import add_timing_noise, relative_times, summary_metrics
    from model import predict, graph_fingerprint
    from signal_story import observe_event
    if model.fingerprint != graph_fingerprint(net):
        raise ValueError('The model does not match the network, sensor order or source domain.')
    held = splits['test']
    if not np.asarray(net.source_pipe_mask)[np.asarray(held['pipe'], dtype=int)].all():
        raise ValueError('The held-out split contains ineligible source pipes.')
    cases = {'Graph-prior held-out sources': (held['pipe'], held['xi'], relative_times(
        add_timing_noise(held['times'], sigma, quantum, np.random.default_rng(seed))))}
    tsdata = load_test_set(net, splits=splits)
    for label, noise, jitter in [('TSNet signals: clean', 0., 0.),
                                 ('TSNet signals: noise and picking uncertainty', pressure_noise_std_m, extra_jitter_ms)]:
        observations = np.stack([observe_event(net, tsdata, index=i, noise_std_m=noise,
                                                jitter_ms=jitter, seed=seed)['relative_s']
                                 for i in range(len(tsdata['pipe']))])
        cases[label] = (tsdata['pipe'], tsdata['xi'], observations)
    report = {'sensors': list(net.sensors), 'min_source_pipe_length_m': float(net.min_source_pipe_length),
              'network_pipes': net.P, 'eligible_source_pipes': net.source_pipe_count,
              'tsnet_source_selection': json.loads(tsdata['source_selection_json']),
              'training': dict(model.training_metadata),
              'evaluation': {'timestamp_sigma_s':sigma, 'timestamp_quantum_s':quantum,
                             'pressure_noise_std_m':pressure_noise_std_m,
                             'extra_jitter_ms':extra_jitter_ms, 'seed':seed,
                             'tsnet_arrivals':'Re-picked from stored 100 Hz signals; no source labels used by detector.'},
              'results':{}}
    arrays = {}
    for case_index,(label,(pipe,xi,observations)) in enumerate(cases.items()):
        valid = np.isfinite(observations).all(axis=1)
        pp = np.full(len(pipe), -1, dtype=int); px = np.full(len(pipe), np.nan)
        probabilities = np.full((len(pipe), net.P), np.nan)
        if valid.any():
            prediction = predict(model, observations[valid])
            pp[valid],px[valid] = prediction['pipe'],prediction['xi']
            probabilities[valid] = prediction['probabilities']
        metrics = {'Pipe accuracy (%)':float(100*np.mean(pp==pipe)),
                   'Mean network distance error (m)':None}
        if valid.all():
            metrics = summary_metrics(net,pipe,xi,pp,px)
        report['results'][label] = {'events':len(pipe), 'localised_events':int(valid.sum()),
                                     'missing_arrival_events':int((~valid).sum()), 'GNN':metrics}
        if not valid.all():
            report['results'][label]['distance_note'] = 'Not defined for the complete set because one or more events have missing arrivals.'
        prefix = ('gttp','tsnet_clean','tsnet_noisy')[case_index]
        arrays.update({prefix+'_observations':observations,prefix+'_true_pipe':pipe,
                       prefix+'_true_xi':xi,prefix+'_gnn_pipe':pp,prefix+'_gnn_xi':px,
                       prefix+'_probabilities':probabilities,prefix+'_detected':valid})
    if output_dir is not None:
        destination = Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
        (destination/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        _atomic_npz(destination/'predictions.npz',**arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--simulate',action='store_true')
    parser.add_argument('--force-simulate',action='store_true',help='Regenerate cached simulations; use with --simulate.')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--events',type=int,default=21)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--min-source-pipe-length',type=float,default=100.)
    parser.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    parser.add_argument('--model',type=Path,default=None)
    args = parser.parse_args()
    if not any([args.train,args.simulate,args.evaluate]):
        parser.error('Choose --train, --simulate and/or --evaluate.')
    from graph_tools import load_network, make_source_split
    from model import train_model,save_model,load_model
    net = load_network(min_source_pipe_length=args.min_source_pipe_length)
    splits = make_source_split(net)
    if args.simulate:
        simulate_test_set(net,splits,args.events,args.workers,force=args.force_simulate)
    if args.train:
        model,history = train_model(net,splits['train'],splits['val'],epochs=args.epochs,
                                    sigma=.010,quantum=.010,patience=30,device=args.device,
                                    checkpoint_path=ROOT/'outputs/training_last.pt')
        model_path = args.model or ROOT/'outputs/gnn_retrained.npz'
        save_model(model,model_path,history)
        print(f'Saved trained model: {model_path}')
    if args.evaluate:
        if not args.train:
            model = load_model(net,args.model or ROOT/'models/gnn_demo.npz',device=args.device)
        report = evaluate(net,splits,model,output_dir=ROOT/'outputs')
        print(json.dumps(report['results'],indent=2))


if __name__ == '__main__':
    main()
