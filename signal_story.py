"""Signal observations for the tutorial's passive-event story.

The recorded waveforms come from TSNet. Noise and arrival picking operate on
the recordings only. Source coordinates and the simulated event onset are
used separately for explanatory overlays and scoring, never to pick arrivals.
"""
from __future__ import annotations

import json
import numpy as np

from tsnet_tools import pick_arrivals

BASELINE_END_S = .3
MIN_PICK_TIME_S = .3


def _sensor_arrays(net, dataset, index=0):
    i = int(index)
    if i != index or not 0 <= i < len(dataset['signal']):
        raise ValueError('Choose an event index present in the TSNet dataset.')
    names = list(np.asarray(dataset['node_names']).astype(str))
    missing = [name for name in net.sensors if name not in names]
    if missing:
        raise ValueError(f'The saved recordings do not contain sensors: {missing}')
    columns = np.asarray([names.index(name) for name in net.sensors], dtype=int)
    time = np.asarray(dataset['time'], dtype=float)
    response = np.asarray(dataset['signal'][i], dtype=float)[:, columns]
    if response.shape != (len(time), len(columns)):
        raise ValueError('TSNet signal dimensions do not match its timestamps and sensor names.')
    if not np.isfinite(response).all():
        raise ValueError('Recorded signals must be finite.')
    return i, time, columns, response


def event_record(net, dataset, index=0):
    """Read one event for display, including separately labelled simulation truth."""
    i, time, columns, response = _sensor_arrays(net, dataset, index)
    heads = np.asarray(dataset['heads'][i], dtype=float)[:, columns]
    elevations = np.asarray(dataset['elevations'], dtype=float)[columns]
    metadata = json.loads(str(np.asarray(dataset['metadata_json']).item()))[i]
    return {'index': i, 'time': time, 'signal': response,
            'pressure_head': heads-elevations[None, :],
            'sensors': tuple(net.sensors), 'columns': columns,
            'pipe': int(dataset['pipe'][i]), 'xi': float(dataset['xi'][i]),
            'onset_s': float(metadata['start_time_s']), 'metadata': metadata,
            'sampling_hz': float(1/np.median(np.diff(time)))}


def observe_event(net, dataset, index=0, noise_std_m=0., jitter_ms=0., seed=42):
    """Add pressure noise, repick arrivals, then optionally add timing jitter.

    ``noise_std_m`` is independent Gaussian pressure-head noise per recorded
    sample and sensor. ``jitter_ms`` is the standard deviation of an additional
    independent Gaussian timestamp error per sensor, added after picking.
    Both streams are reproducible; changing jitter leaves the noisy signal
    unchanged. No source coordinates, onset time or theoretical arrival enter
    this function. Missing detections remain NaN and block GNN inference.

    The compact cache is sampled at 100 Hz. We rerun the detector on those
    stored records, then round timestamps to that sampling grid. The optional
    linear threshold interpolation belongs to the picker; the final features
    use the explicit sampling resolution.
    """
    noise_std_m, jitter_ms = float(noise_std_m), float(jitter_ms)
    if not np.isfinite([noise_std_m,jitter_ms]).all() or min(noise_std_m,jitter_ms) < 0:
        raise ValueError('Noise and additional picking-error standard deviations must be finite and nonnegative.')
    i, time, columns, clean = _sensor_arrays(net, dataset, index)
    sequence = np.random.SeedSequence([int(seed), i])
    signal_seed, jitter_seed = sequence.spawn(2)
    noise = np.random.default_rng(signal_seed).normal(0., noise_std_m, clean.shape)
    noisy = clean+noise
    picked, details = pick_arrivals(time, noisy, baseline_end=BASELINE_END_S,
                                    min_time=MIN_PICK_TIME_S, return_details=True)
    clean_picked = pick_arrivals(time, clean, baseline_end=BASELINE_END_S,
                                 min_time=MIN_PICK_TIME_S)
    extra = np.random.default_rng(jitter_seed).normal(0., jitter_ms/1000., len(net.sensors))
    quantum = float(np.median(np.diff(time)))
    if quantum <= 0 or not np.allclose(np.diff(time), quantum, rtol=1e-7, atol=1e-10):
        raise ValueError('The tutorial signal cache must use a regular sampling grid.')
    shifted = picked+extra
    arrivals = np.sign(shifted)*np.floor(np.abs(shifted)/quantum+.5)*quantum
    detected = np.isfinite(arrivals)
    relative = arrivals-arrivals.min() if detected.all() else np.full_like(arrivals,np.nan)
    return {'index':i, 'time':time, 'clean_signal':clean, 'noisy_signal':noisy,
            'noise':noise, 'noise_std_m':noise_std_m, 'jitter_ms':jitter_ms,
            'clean_picks_s':clean_picked, 'signal_picks_s':picked,
            'extra_jitter_s':extra, 'arrivals_s':arrivals,
            'relative_s':relative, 'detected':detected, 'details':details,
            'quantum_s':quantum, 'seed':int(seed), 'sensors':tuple(net.sensors)}


def graph_arrivals(net, dataset, index=0):
    """Known-source reference for teaching; this is not an inference input."""
    from graph_tools import travel_times
    event = event_record(net, dataset, index)
    return event['onset_s']+travel_times(net,[event['pipe']],[event['xi']])[0]


def arrival_table(net, dataset, index=0, noise_std_m=0., jitter_ms=0., seed=42):
    """Explain predictions, signal picks and observations in separate columns."""
    import pandas as pd
    obs = observe_event(net,dataset,index,noise_std_m,jitter_ms,seed)
    expected = graph_arrivals(net,dataset,index)
    return pd.DataFrame({
        'Sensor':list(net.sensors),
        'Graph arrival (s)':expected,
        'Signal pick (s)':obs['signal_picks_s'],
        'Extra pick error (ms)':1000*obs['extra_jitter_s'],
        'Observation on grid (s)':obs['arrivals_s'],
        'Relative observation (s)':obs['relative_s'],
        'Signal pick minus graph (ms)':1000*(obs['signal_picks_s']-expected),
        'Detected':obs['detected'],
    })


def localise_observation(net, model, observation):
    """Run the GNN only when every selected sensor has a valid arrival."""
    if not np.asarray(observation['detected']).all():
        missing = [s for s,ok in zip(net.sensors,observation['detected']) if not ok]
        return {'available':False,'message':'No arrival detected at '+', '.join(missing)+'.'}
    from model import predict
    result = predict(model,np.asarray(observation['relative_s'])[None,:])
    return {'available':True,'pipe':int(result['pipe'][0]),'xi':float(result['xi'][0]),
            'probabilities':np.asarray(result['probabilities'][0])}
