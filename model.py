"""A small graph neural network for a fixed, instrumented pipe network.

Two outputs describe one location: a pipe probability and a fraction along it.
The encoder shares weights across nodes; the decoder shares weights across
pipes. Graph edges actively mix neighbouring node representations.

For a large graph, local sensor readings alone need many message-passing steps
to reach distant nodes. We therefore broadcast the measured sensor timing
vector to each node and supply known node-to-sensor travel-time landmarks.
Both are available at inference. Neither contains the unknown source position.
This is an explicit feature-design choice, not a claim that two graph layers
can move information across the entire network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _source_pipe_mask(net):
    """Eligible source pipes in the original, unchanged pipe indexing."""
    threshold = float(getattr(net, 'min_source_pipe_length', 0.))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('The minimum source-pipe length must be finite and nonnegative.')
    mask = np.asarray(getattr(net, 'source_pipe_mask',
                              np.asarray(net.lengths) >= threshold), dtype=bool)
    if mask.shape != (len(net.pipe_names),) or not mask.any():
        raise ValueError('At least one original pipe must be eligible as a source.')
    return mask


def graph_fingerprint(net):
    """Protect the network, sensor ordering and source-pipe eligibility."""
    descriptor = {
        'nodes': list(net.names), 'pipes': list(net.pipe_names),
        'ends': np.asarray(net.ends).tolist(),
        'lengths': np.asarray(net.lengths).round(9).tolist(),
        'diameters': np.asarray(net.diameters).round(9).tolist(),
        'positions': np.asarray(net.positions).round(9).tolist(),
        'adjacency': np.asarray(net.adjacency).astype(int).tolist(),
        'pipe_wave_speeds': np.asarray(getattr(net, 'pipe_wave_speeds', [])).round(9).tolist(),
        'nominal_pipe_wave_speeds': np.asarray(net.nominal_pipe_wave_speeds).round(9).tolist(),
        'reference_dt': round(float(getattr(net, 'reference_dt', 0.)), 12),
        'sensors': list(net.sensors),
        'travel_distances': np.nan_to_num(np.asarray(net.travel_distances),
                                          posinf=-999.).round(9).tolist(),
    }
    source_mask = _source_pipe_mask(net)
    threshold = float(getattr(net, 'min_source_pipe_length', 0.))
    # Keep the original all-pipe fingerprint so supplied checkpoints still load.
    if threshold > 0 or not source_mask.all():
        descriptor['min_source_pipe_length'] = threshold
        descriptor['source_pipe_indices'] = np.flatnonzero(source_mask).tolist()
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


class GraphSourceGNN(nn.Module):
    """Residual message passing followed by a shared two-output edge decoder.

    Input: (number of events, number of sensors), relative arrivals in seconds.
    Output: pipe logits and within-pipe fractions, both (events, pipes).
    Fractions are measured from each pipe's original EPANET start node.
    Ineligible source pipes retain their columns but receive zero probability.
    They remain part of the graph used for message passing and wave travel.
    """

    def __init__(self, net, hidden=32, layers=2):
        super().__init__()
        self.architecture = 'residual_gcn'
        self.position_mode = 'sigmoid'
        self.hidden, self.layers = int(hidden), int(layers)
        self.fingerprint = graph_fingerprint(net)
        self.network_contract = _network_contract(net)
        self.sensor_names = list(net.sensors)
        self.pipe_names = list(net.pipe_names)
        self.node_names = list(net.names)
        self.training_metadata = {}
        self.min_source_pipe_length = float(getattr(net, 'min_source_pipe_length', 0.))
        source_mask = _source_pipe_mask(net)
        self.source_pipe_count = int(source_mask.sum())
        # Rebuild from the fingerprint-checked network on load. This avoids
        # changing the state_dict keys of existing all-pipe checkpoints.
        self.register_buffer('source_pipe_mask', torch.tensor(source_mask, dtype=torch.bool),
                             persistent=False)

        distance = np.asarray(net.travel_distances, dtype=np.float32)
        boundary = ~np.isfinite(distance).all(axis=1)
        finite = distance[np.isfinite(distance)]
        self.time_scale = float(max(np.quantile(finite, 0.90), 0.1))
        # Fixed-head storage nodes have no source-to-sensor timing signature.
        # A boundary flag makes their zero-filled landmark row unambiguous.
        landmarks = np.where(np.isfinite(distance), distance, 0.)
        landmarks = (landmarks - landmarks.mean(axis=1, keepdims=True)) / self.time_scale
        xy = np.asarray(net.positions, dtype=np.float32)
        xy = (xy - xy.mean(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1.)
        adjacency = np.asarray(net.adjacency, dtype=np.float32)
        degree = adjacency.sum(axis=1)
        with_self = adjacency + np.eye(len(adjacency), dtype=np.float32)
        invroot = 1. / np.sqrt(with_self.sum(axis=1))
        normalized = invroot[:, None] * with_self * invroot[None, :]
        mask = np.zeros(len(adjacency), dtype=np.float32)
        mask[np.asarray(net.sensor_indices)] = 1.
        static = np.column_stack((xy, degree / max(degree.max(), 1.), boundary, mask)).astype(np.float32)

        self.register_buffer('landmarks', torch.tensor(landmarks, dtype=torch.float32))
        self.register_buffer('static', torch.tensor(static))
        self.register_buffer('normalized_adjacency', torch.tensor(normalized))
        self.register_buffer('sensor_indices', torch.tensor(net.sensor_indices, dtype=torch.long))
        self.register_buffer('ends', torch.tensor(net.ends, dtype=torch.long))
        lengths = np.asarray(net.lengths, dtype=np.float32)
        diameters = np.asarray(net.diameters, dtype=np.float32)
        edge_static = np.column_stack((lengths / max(lengths.max(), 1.),
                                       diameters / max(diameters.max(), 0.01)))
        self.register_buffer('edge_static', torch.tensor(edge_static, dtype=torch.float32))

        # Landmarks, observed timings, their residual and its square, plus
        # coordinates/degree/boundary/sensor flags and the local timing value.
        sensor_count = len(net.sensors)
        self.input_layer = nn.Linear(4 * sensor_count + 6, self.hidden)
        self.self_layers = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.layers)])
        self.neighbour_layers = nn.ModuleList([nn.Linear(self.hidden, self.hidden, bias=False)
                                              for _ in range(self.layers)])
        edge_dim = 3 * self.hidden + 3 * sensor_count + 2
        self.edge_decoder = nn.Sequential(nn.Linear(edge_dim, 64), nn.GELU(),
                                          nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    def input_features(self, relative_arrivals):
        """Supply explicit hydraulic features before their learned transformation."""
        if relative_arrivals.ndim != 2 or relative_arrivals.shape[1] != len(self.sensor_names):
            raise ValueError('Expected an events-by-sensors array in the saved sensor order.')
        # train_model/predict validate inputs before transferring them. A
        # tensor-to-Python finite check here would synchronise every GPU batch.
        observed = relative_arrivals - relative_arrivals.mean(dim=1, keepdim=True)
        observed = observed / self.time_scale
        batch, nodes = len(observed), len(self.node_names)
        obs = observed[:, None, :].expand(-1, nodes, -1)
        landmarks = self.landmarks[None].expand(batch, -1, -1)
        residual = obs - landmarks
        local = observed.new_zeros((batch, nodes, 1))
        local[:, self.sensor_indices, 0] = observed
        features = torch.cat((landmarks, obs, residual, residual.square(),
                              self.static[None].expand(batch, -1, -1), local), dim=-1)
        return features

    def encode(self, relative_arrivals):
        """Return the initial embedding and every message-passing embedding."""
        features = self.input_features(relative_arrivals)
        observed = (relative_arrivals-relative_arrivals.mean(dim=1, keepdim=True))/self.time_scale
        h = F.gelu(self.input_layer(features))
        history = [h]
        for own, neighbour in zip(self.self_layers, self.neighbour_layers):
            aggregated = torch.matmul(self.normalized_adjacency, h)
            h = F.gelu(own(h) + neighbour(aggregated)) + 0.2 * h
            history.append(h)
        return history, observed

    def forward(self, relative_arrivals, return_embeddings=False):
        embeddings, observed = self.encode(relative_arrivals)
        h = embeddings[-1]
        u, v = self.ends[:, 0], self.ends[:, 1]
        hu, hv = h[:, u], h[:, v]
        batch, pipes = len(h), len(u)
        evidence = torch.cat((self.landmarks[u], self.landmarks[v]), dim=-1)
        edge_features = torch.cat((hu, hv, torch.abs(hu-hv),
                                   observed[:, None, :].expand(-1, pipes, -1),
                                   evidence[None].expand(batch, -1, -1),
                                   self.edge_static[None].expand(batch, -1, -1)), dim=-1)
        raw = self.edge_decoder(edge_features)
        logits = raw[..., 0].masked_fill(~self.source_pipe_mask[None, :], float('-inf'))
        result = (logits, torch.sigmoid(raw[..., 1]))
        if return_embeddings:
            return (*result, embeddings)
        return result


def noisy_relative(times, sigma=0.010, quantum=0.010, rng=None, *,
                   picking_error_samples=None, sampling_hz=100.0):
    """Add noise to RAW arrivals, quantize, then remove the unknown onset.

    Taking the differences last preserves the shared-reference noise structure.
    sigma and quantum are seconds, not fractions of a pipe travel time.
    With picking_error_samples=(minimum, maximum), sigma must be zero and
    independent signed integer sample errors replace Gaussian jitter.
    """
    from graph_tools import add_timing_noise
    noisy = add_timing_noise(times, sigma, quantum, rng,
                             picking_error_samples=picking_error_samples,
                             sampling_hz=sampling_hz)
    return (noisy - noisy.min(axis=1, keepdims=True)).astype(np.float32)


def _batch_loss(model, x, pipe, xi, position_weight):
    logits, positions = model(x)
    ce = F.cross_entropy(logits, pipe)
    selected = positions[torch.arange(len(pipe), device=pipe.device), pipe]
    mse = F.mse_loss(selected, xi)
    return ce + position_weight * mse, ce, mse, logits.argmax(dim=1)


def _resolve_device(device='auto'):
    """Probe a real CUDA operation and fall back to CPU if it is unusable."""
    import warnings
    if str(device).lower() in ('auto', 'cuda') or str(device).lower().startswith('cuda:'):
        if torch.cuda.is_available():
            try:
                target = torch.device('cuda' if str(device).lower()=='auto' else device)
                # Querying availability alone does not detect unsupported GPU kernels.
                probe = torch.ones((2, 2), device=target)
                (probe @ probe).sum().item()
                return target
            except (RuntimeError, AssertionError) as error:
                warnings.warn(f'CUDA could not execute a tensor operation; continuing on CPU. {error}', RuntimeWarning)
        elif str(device).lower() != 'auto':
            warnings.warn('CUDA is unavailable in this environment; continuing on CPU.', RuntimeWarning)
        return torch.device('cpu')
    target = torch.device(device)
    if target.type != 'cpu':
        raise ValueError("This tutorial supports device='auto', 'cuda' or 'cpu'.")
    return target


def _cpu_tree(value):
    """Detach tensor state for portable, independent checkpoints."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sources_fingerprint(train_sources, val_sources):
    digest = hashlib.sha256()
    for label, source in [('train', train_sources), ('validation', val_sources)]:
        digest.update(label.encode())
        for key in ['times', 'pipe', 'xi']:
            array = np.ascontiguousarray(source[key], dtype=np.int64 if key == 'pipe' else np.float64)
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def train_model(net, train_sources, val_sources, epochs=500, sigma=0.010,
                quantum=0.010, seed=42, batch_size=128, hidden=32, layers=2,
                learning_rate=0.002, position_weight=4., patience=30,
                verbose=True, model=None, threads=None, device='auto',
                checkpoint_path='outputs/training_last.pt', resume=False, print_every=10,
                picking_error_samples=None, sampling_hz=100.0):
    """Train on CUDA when available, with optional exact epoch-boundary resume.

    ``epochs`` is the TOTAL epoch limit, including previously completed epochs.
    The checkpoint stores the last completed epoch (weights, optimiser,
    scheduler and RNG state) separately from the best validation weights.
    Interrupting a partial epoch rolls back to the last completed checkpoint.
    The returned model always contains the best completed validation weights.
    Set sigma=0 and picking_error_samples=(2, 10) for independent signed
    2--10 sample picking errors at sampling_hz. Training draws refresh each
    epoch; validation draws remain fixed. Checkpoints record the noise model.
    """
    from graph_tools import timing_noise_config
    if int(epochs) < 1 or int(batch_size) < 1 or int(patience) < 1 or int(print_every) < 1:
        raise ValueError('epochs, batch_size, patience and print_every must be positive.')
    timing_noise = timing_noise_config(sigma, quantum, picking_error_samples, sampling_hz)
    sigma, quantum = timing_noise['sigma_s'], timing_noise['quantum_s']
    sampling_hz = timing_noise['sampling_hz']
    if picking_error_samples is not None:
        picking_error_samples = (timing_noise['min_samples'], timing_noise['max_samples'])
    noise_options = dict(picking_error_samples=picking_error_samples,
                         sampling_hz=sampling_hz)
    if resume and checkpoint_path is None:
        raise ValueError('resume=True requires checkpoint_path.')
    source_mask = _source_pipe_mask(net)
    for source in [train_sources, val_sources]:
        count = len(source['pipe'])
        if (np.asarray(source['times']).shape != (count, net.S)
                or np.asarray(source['xi']).shape != (count,)):
            raise ValueError('Each source needs one pipe, one fraction and a timing vector in sensor order.')
        if np.any((np.asarray(source['xi']) <= 0) | (np.asarray(source['xi']) >= 1)):
            raise ValueError('Training source fractions must be strictly inside their original pipes.')
        if len(source['pipe']) == 0 or any(not np.isfinite(source[key]).all()
                                          for key in ['times', 'pipe', 'xi']):
            raise ValueError('Training and validation sources must be nonempty and finite.')
        pipe_ids = np.asarray(source['pipe'])
        if (pipe_ids.ndim != 1 or np.any(pipe_ids < 0)
                or np.any(pipe_ids >= len(source_mask))
                or np.any(pipe_ids != np.floor(pipe_ids))):
            raise ValueError('Source pipe labels must be integer indices into the original network.')
        if not source_mask[pipe_ids.astype(np.int64)].all():
            raise ValueError('Training or validation contains an ineligible source pipe. '
                             'Regenerate the source splits with the current minimum pipe length.')
    train_locations = set(zip(np.asarray(train_sources['pipe']).tolist(), np.asarray(train_sources['xi']).tolist()))
    val_locations = set(zip(np.asarray(val_sources['pipe']).tolist(), np.asarray(val_sources['xi']).tolist()))
    if train_locations & val_locations:
        raise ValueError('Training and validation source locations must be disjoint.')
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(position_weight) or position_weight < 0:
        raise ValueError('Use a positive finite learning rate and nonnegative finite position weight.')
    target_device = _resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if model is None:
        model = GraphSourceGNN(net, hidden=hidden, layers=layers)
    if model.fingerprint != graph_fingerprint(net):
        raise ValueError('The supplied model does not match this graph, sensor set '
                         'or source-pipe eligibility. Start a fresh model after changing the length threshold.')
    model.to(target_device)
    cpu_threads = min(2, os.cpu_count() or 1) if threads is None else int(threads)
    if cpu_threads < 1:
        raise ValueError('threads must be positive.')
    torch.set_num_threads(cpu_threads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    settings = dict(sigma=float(sigma), quantum=float(quantum), seed=int(seed),
                    batch_size=int(batch_size), hidden=model.hidden, layers=model.layers,
                    learning_rate=float(learning_rate), position_weight=float(position_weight),
                    patience=int(patience), sources_fingerprint=_sources_fingerprint(train_sources, val_sources))
    # Preserve legacy Gaussian checkpoint settings exactly. Sample-error
    # checkpoints additionally guard the distribution and sampling frequency.
    if picking_error_samples is not None:
        settings['timing_noise'] = timing_noise
    history = {'epoch': [], 'train_loss': [], 'val_loss': [],
               'train_pipe_accuracy': [], 'val_pipe_accuracy': [],
               'val_position_mse': [], 'learning_rate': [], 'epoch_seconds': []}
    best_loss, best_state, best_epoch, completed_epoch = float('inf'), None, 0, 0
    last_record = None
    initial_state = _cpu_tree(model.state_dict())
    if resume:
        last_record = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if last_record.get('format_version') != 2:
            raise ValueError('This is not a resumable training checkpoint. Use the training-state file.')
        if last_record['fingerprint'] != model.fingerprint or last_record['settings'] != settings:
            raise ValueError('Resume settings or source splits differ from the checkpoint. '
                             'Restore them, or start a fresh run with resume=False.')
        model.load_state_dict(last_record['last_state'])
        optimizer.load_state_dict(last_record['optimizer_state'])
        scheduler.load_state_dict(last_record['scheduler_state'])
        history = copy.deepcopy(last_record['history'])
        history.pop('interrupted', None)
        best_state = _cpu_tree(last_record['best_state'])
        best_loss = float(last_record['best_loss'])
        best_epoch, completed_epoch = int(last_record['best_epoch']), int(last_record['epoch'])
        rng.bit_generator.state = json.loads(last_record['numpy_rng_state_json'])
        torch.set_rng_state(last_record['torch_rng_state'])
        if target_device.type == 'cuda' and last_record.get('cuda_rng_states'):
            for index, state in enumerate(last_record['cuda_rng_states'][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, index)
    y_train = torch.as_tensor(train_sources['pipe'], dtype=torch.long, device=target_device)
    z_train = torch.as_tensor(train_sources['xi'], dtype=torch.float32, device=target_device)
    x_val = torch.as_tensor(noisy_relative(val_sources['times'], sigma, quantum,
                                          np.random.default_rng(seed + 1),
                                          **noise_options), device=target_device)
    y_val = torch.as_tensor(val_sources['pipe'], dtype=torch.long, device=target_device)
    z_val = torch.as_tensor(val_sources['xi'], dtype=torch.float32, device=target_device)
    if verbose:
        name = torch.cuda.get_device_name(target_device) if target_device.type == 'cuda' else f'CPU ({cpu_threads} threads)'
        print(f'Training device: {name}; batch size: {batch_size}; total epoch limit: {epochs}', flush=True)
        if resume:
            print(f'Resuming after completed epoch {completed_epoch}; best validation epoch {best_epoch}.', flush=True)
        if checkpoint_path is not None:
            print(f'Completed-epoch checkpoints: {Path(checkpoint_path)}', flush=True)
    began = time.monotonic()
    interrupted = False
    already_stopped = completed_epoch > 0 and completed_epoch - best_epoch >= patience
    if already_stopped and verbose:
        print('This checkpoint already met the early-stopping condition; returning its best model.', flush=True)
    try:
        for epoch in range(completed_epoch + 1, int(epochs) + 1) if not already_stopped else []:
            epoch_began = time.monotonic()
            model.train()
            # One transfer per epoch, rather than transferring every mini-batch.
            x_train = torch.as_tensor(noisy_relative(train_sources['times'], sigma, quantum, rng,
                                                    **noise_options),
                                      device=target_device)
            order = torch.as_tensor(rng.permutation(len(y_train)), dtype=torch.long, device=target_device)
            train_totals = torch.zeros(2, device=target_device)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, _, _, pred = _batch_loss(model, x_train[indices], y_train[indices],
                                               z_train[indices], position_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                train_totals[0] += loss.detach() * len(indices)
                train_totals[1] += (pred == y_train[indices]).sum().detach()
            model.eval()
            val_totals = torch.zeros(3, device=target_device)
            with torch.no_grad():
                for start in range(0, len(y_val), batch_size):
                    stop = start + batch_size
                    loss, _, mse, pred = _batch_loss(model, x_val[start:stop], y_val[start:stop],
                                                    z_val[start:stop], position_weight)
                    val_totals[0] += loss.detach() * len(pred)
                    val_totals[1] += (pred == y_val[start:stop]).sum()
                    val_totals[2] += mse.detach() * len(pred)
            # Synchronise just the five summary values once per epoch.
            numbers = torch.cat((train_totals, val_totals)).cpu().numpy()
            validation_loss = float(numbers[2] / len(y_val))
            scheduler.step(validation_loss)
            history['epoch'].append(epoch)
            history['train_loss'].append(float(numbers[0] / len(y_train)))
            history['val_loss'].append(validation_loss)
            history['train_pipe_accuracy'].append(float(numbers[1] / len(y_train)))
            history['val_pipe_accuracy'].append(float(numbers[3] / len(y_val)))
            history['val_position_mse'].append(float(numbers[4] / len(y_val)))
            history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
            history['epoch_seconds'].append(float(time.monotonic() - epoch_began))
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = _cpu_tree(model.state_dict())
            record = dict(format_version=2, fingerprint=model.fingerprint, settings=settings,
                          epoch=epoch, last_state=_cpu_tree(model.state_dict()),
                          optimizer_state=_cpu_tree(optimizer.state_dict()),
                          scheduler_state=_cpu_tree(scheduler.state_dict()), history=copy.deepcopy(history),
                          best_state=best_state, best_loss=best_loss, best_epoch=best_epoch,
                          numpy_rng_state_json=json.dumps(rng.bit_generator.state),
                          torch_rng_state=torch.get_rng_state().cpu(),
                          cuda_rng_states=[state.cpu() for state in torch.cuda.get_rng_state_all()]
                          if target_device.type == 'cuda' else None)
            if checkpoint_path is not None:
                _atomic_save(record, checkpoint_path)
            last_record, completed_epoch = record, epoch
            if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                print(f'Epoch {epoch:4d}/{epochs}: train loss {history["train_loss"][-1]:.3f}, '
                      f'validation loss {validation_loss:.3f}, '
                      f'validation pipe accuracy {100*history["val_pipe_accuracy"][-1]:.1f}%, '
                      f'{history["epoch_seconds"][-1]:.1f} s/epoch', flush=True)
            if epoch - best_epoch >= patience:
                if verbose:
                    print(f'Early stopping; best validation checkpoint: epoch {best_epoch}.', flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        if last_record is None:
            model.load_state_dict(initial_state)
            print('Interrupted before the first completed epoch; no resumable epoch was saved.', flush=True)
            raise
        # Do not pair partial-epoch model weights with an older optimiser/RNG.
        history = copy.deepcopy(last_record['history'])
        best_state = last_record['best_state']
        best_loss, best_epoch = last_record['best_loss'], last_record['best_epoch']
        completed_epoch = last_record['epoch']
        print(f'Interrupted. Discarded the partial epoch; returning the best model through epoch {completed_epoch}.', flush=True)
        if checkpoint_path is not None:
            print('Set resume=True with the same checkpoint_path to continue from the last completed epoch.', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    history['best_epoch'] = int(best_epoch)
    history['completed_epochs'] = int(completed_epoch)
    history['interrupted'] = interrupted
    history['elapsed_seconds_this_call'] = float(time.monotonic() - began)
    model.training_metadata = dict(seed=seed, sigma=sigma, quantum=quantum,
                                   timing_noise=timing_noise,
                                   architecture='residual_gcn', position_mode='sigmoid',
                                   position_weight=position_weight, best_epoch=int(best_epoch),
                                   completed_epochs=int(completed_epoch), training_device=str(target_device),
                                   train_sources=len(y_train), validation_sources=len(y_val),
                                   min_source_pipe_length=model.min_source_pipe_length,
                                   source_pipe_count=model.source_pipe_count,
                                   validation_loss=float(best_loss), input='relative first-arrival times',
                                   features='broadcast timings + graph travel-time landmarks + local timing/mask')
    model.history = history
    return model, history


def predict(model, relative_arrivals, return_embeddings=False, batch_size=128):
    """Predict locations; rows must follow model.sensor_names exactly."""
    x = np.asarray(relative_arrivals, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Arrival inputs must be a nonempty, finite events-by-sensors array.')
    device = next(model.parameters()).device
    model.eval()
    probabilities, fractions, all_embeddings = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            output = model(torch.as_tensor(x[start:start + batch_size], device=device),
                           return_embeddings=return_embeddings)
            logits, positions = output[:2]
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            fractions.append(positions.cpu().numpy())
            if return_embeddings:
                all_embeddings.append([h.cpu().numpy() for h in output[2]])
    probabilities, fractions = np.concatenate(probabilities), np.concatenate(fractions)
    pipe = probabilities.argmax(axis=1)
    result = dict(pipe=pipe, xi=fractions[np.arange(len(pipe)), pipe],
                  probabilities=probabilities, positions=fractions)
    if return_embeddings:
        result['embeddings'] = [np.concatenate([b[layer] for b in all_embeddings])
                                for layer in range(model.layers + 1)]
    return result


def _network_contract(net):
    """Ordered physical inputs and derived features stored with portable weights."""
    values = {'nodes': np.asarray(net.names), 'pipes': np.asarray(net.pipe_names),
            'sensors': np.asarray(net.sensors), 'ends': np.asarray(net.ends, dtype=np.int64),
            'sensor_indices': np.asarray(net.sensor_indices, dtype=np.int64),
            'lengths': np.asarray(net.lengths, dtype=np.float64),
            'diameters': np.asarray(net.diameters, dtype=np.float64),
            'positions': np.asarray(net.positions, dtype=np.float64),
            'adjacency': np.asarray(net.adjacency, dtype=np.int64),
            'pipe_wave_speeds': np.asarray(net.pipe_wave_speeds, dtype=np.float64),
            'nominal_pipe_wave_speeds': np.asarray(net.nominal_pipe_wave_speeds, dtype=np.float64),
            'reference_dt': np.asarray(net.reference_dt, dtype=np.float64),
            'travel_distances': np.asarray(net.travel_distances, dtype=np.float64),
            'source_pipe_indices': np.asarray(net.source_pipe_indices, dtype=np.int64),
            'min_source_pipe_length': np.asarray(net.min_source_pipe_length, dtype=np.float64)}
    return {key: value.copy() for key,value in values.items()}


def _check_contract(net, saved):
    current = _network_contract(net)
    for name, value in current.items():
        expected = saved[name]
        same = value.shape == expected.shape
        if same and value.dtype.kind in 'f':
            # Physical dimensions must agree within tiny numerical roundoff;
            # ordered names, sensors, topology and eligibility are exact.
            same = np.allclose(value, expected, rtol=1e-11, atol=1e-9)
        elif same:
            same = np.array_equal(value, expected)
        if not same:
            raise ValueError(f'Saved model is incompatible with {name}. Restore its network, sensor and source settings, or retrain.')


def save_model(model, path, history=None):
    """Write portable NumPy weights and JSON metadata, without Python pickles."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Save inference models with a .npz extension.')
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'format_version': 3, 'architecture': 'residual_gcn',
                'position_mode': 'sigmoid', 'hidden': model.hidden, 'layers': model.layers,
                'training_metadata': model.training_metadata,
                'history': history if history is not None else getattr(model, 'history', None)}
    arrays = {'state__'+key: value.detach().cpu().numpy() for key,value in model.state_dict().items()}
    arrays.update({'contract__'+key: value for key,value in model.network_contract.items()})
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(net, path, device='auto'):
    """Load portable weights only after validating graph and sensor features."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Use the supplied models/gnn_demo.npz or a newly trained .npz model.')
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        if metadata.get('format_version') != 3 or metadata.get('architecture') != 'residual_gcn':
            raise ValueError('This is not a supported tutorial GNN checkpoint.')
        _check_contract(net, {key.removeprefix('contract__'): saved[key] for key in saved.files if key.startswith('contract__')})
        state = {key.removeprefix('state__'): torch.from_numpy(saved[key].copy()) for key in saved.files if key.startswith('state__')}
    model = GraphSourceGNN(net, hidden=metadata['hidden'], layers=metadata['layers'])
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError('Saved weights do not match the declared GNN dimensions.') from error
    target = _resolve_device(device)
    model.to(target)
    if target.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    model.training_metadata = metadata.get('training_metadata', {})
    model.history = metadata.get('history')
    model.eval()
    return model
