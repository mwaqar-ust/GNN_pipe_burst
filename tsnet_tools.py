"""Real TSNet burst simulations and signal-only arrival picking.

This module is deliberately separate from the graph travel-time generator.
The testing signals come from TSNet's method-of-characteristics solver.
For these controlled numerical experiments, a matched no-burst simulation
removes small initialisation drifts before the arrival detector is applied.
That background subtraction is not a claim of a field-ready detector.
"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import copy
import importlib.metadata
import io
import json
import os
import tempfile
import time as clock
import warnings

import numpy as np
from network_config import (discretize, DEFAULT_DT_S, DEFAULT_DURATION_S,
                            BURST_COEFFICIENT, DARCY_FRICTION_FACTOR)


DEFAULT_INP = Path(__file__).resolve().parent / "networks" / "HKUST.inp"


@contextmanager
def _temporary_working_directory():
    """Isolate EPANET's temporary files; always restore the caller's folder.

    Working directories are process-wide. Run simulations sequentially, or
    use separate processes rather than threads for parallel simulations.
    """
    previous = Path.cwd()
    initialize = importlib.import_module('tsnet.simulation.initialize')
    single = importlib.import_module('tsnet.simulation.single')
    original_roughness, original_leakage = initialize.cal_roughness_coef, single.add_leakage
    def roughness(pipe, velocity, head_loss):
        # The controlled HKUST model starts at rest. A zero-flow initial state
        # cannot identify a friction factor from velocity/head-loss division.
        if abs(velocity) < 1e-5 or head_loss < 1e-3:
            pipe._roughness, pipe.roughness_height = 0., 0.
            return pipe
        return original_roughness(pipe, velocity, head_loss)
    def leakage(*args, **kwargs):
        h, v = original_leakage(*args, **kwargs)
        return float(np.asarray(h).reshape(-1)[0]), float(np.asarray(v).reshape(-1)[0])
    with tempfile.TemporaryDirectory(prefix="iahr_tsnet_") as folder:
        try:
            initialize.cal_roughness_coef, single.add_leakage = roughness, leakage
            os.chdir(folder)
            yield Path(folder)
        finally:
            os.chdir(previous)
            initialize.cal_roughness_coef, single.add_leakage = original_roughness, original_leakage


def pick_arrivals(time, heads, baseline_end=0.3, min_time=0.3,
                  absolute_threshold=.001, noise_multiplier=6.0,
                  relative_threshold=1e-5, persistence=2,
                  return_details=False):
    """Pick the first sustained change using only the supplied signals.

    Parameters
    ----------
    time : (T,) array
        Recorded timestamps, in seconds.
    heads : (T, S) array
        Head traces. Matched background-subtracted responses are recommended
        for this controlled synthetic test. Raw field signals need a detector
        validated for their own drift and noise conditions.
    baseline_end, min_time : float
        A fixed observation lead-in, identical for every event. Neither is
        inferred from the true source or from theoretical travel times.

    Returns
    -------
    (S,) array
        Arrival timestamps. NaN explicitly indicates no detected arrival.

    The threshold combines an absolute floor, robust baseline noise and a
    very small signal-relative floor. Requiring successive threshold crossings
    rejects isolated spikes. It does not use source labels or graph distances.
    """
    time = np.asarray(time, dtype=float).ravel()
    y = np.asarray(heads, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != len(time):
        raise ValueError("heads must have shape (number of timestamps, sensors)")
    if len(time) < 5 or not np.all(np.diff(time) > 0):
        raise ValueError("At least five strictly increasing timestamps are required")
    if not np.all(np.isfinite(y)):
        raise ValueError("The arrival detector does not accept non-finite traces")
    baseline = time <= float(baseline_end)
    if baseline.sum() < 3:
        raise ValueError("The observation lead-in must contain at least three samples")
    if persistence < 1:
        raise ValueError("persistence must be a positive integer")
    level = np.median(y[baseline], axis=0)
    magnitude = np.abs(y - level)
    # MAD is used for a robust estimate of baseline standard deviation.
    sigma = 1.4826 * np.median(np.abs(y[baseline] - level), axis=0)
    peak = np.max(magnitude, axis=0)
    threshold = np.maximum.reduce([
        np.full(y.shape[1], float(absolute_threshold)),
        float(noise_multiplier) * sigma,
        float(relative_threshold) * peak,
    ])
    arrivals = np.full(y.shape[1], np.nan)
    indices = np.full(y.shape[1], -1, dtype=int)
    for j in range(y.shape[1]):
        crosses = (magnitude[:, j] >= threshold[j]) & (time >= min_time)
        runs = np.convolve(crosses.astype(int), np.ones(persistence, dtype=int),
                           mode="valid")
        found = np.flatnonzero(runs == persistence)
        if not len(found):
            continue
        k = int(found[0])
        indices[j] = k
        arrivals[j] = time[k]
        # Linear interpolation estimates the threshold crossing within one
        # sampling interval, without using any theoretical arrival time.
        if k > 0 and magnitude[k, j] > magnitude[k - 1, j]:
            fraction = ((threshold[j] - magnitude[k - 1, j]) /
                        (magnitude[k, j] - magnitude[k - 1, j]))
            arrivals[j] = time[k - 1] + np.clip(fraction, 0, 1) * (time[k] - time[k - 1])
    if return_details:
        return arrivals, {"threshold_m": threshold, "baseline_noise_std_m": sigma,
                          "indices": indices, "detected": np.isfinite(arrivals)}
    return arrivals


def _extract_heads(model, node_names):
    timestamps = np.asarray(model.simulation_timestamps, dtype=float).ravel()
    columns = [np.asarray(model.get_node(n)._head, dtype=float).ravel()
               for n in node_names]
    lengths = {len(timestamps), *[len(v) for v in columns]}
    if len(lengths) != 1:
        raise RuntimeError("TSNet returned inconsistent head and timestamp lengths")
    values = np.column_stack(columns)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("TSNet produced non-finite heads; this event is invalid")
    return timestamps, values


def simulate_burst(pipe_name, xi, sensors=None, *, inp_path=None,
                   start_time=0.5, rise_time=0.02, coefficient=BURST_COEFFICIENT,
                   duration=DEFAULT_DURATION_S, dt=DEFAULT_DT_S, wave_speed=None,
                   background_subtract=True, baseline_end=0.3,
                   friction="steady"):
    """Run a genuine TSNet burst at an interior position of an original pipe.

    ``xi`` is measured from the pipe's original start node. All original
    junctions are recorded, so changing sensor selection does not require
    rerunning the hydraulic solver. The inserted burst junction is excluded
    from ``node_names`` and therefore cannot leak its position to the GNN.

    ``heads`` contains total hydraulic heads. ``signal`` contains the matched
    background-subtracted burst response (or the baseline-centred raw heads
    when background_subtract=False). ``arrivals`` is detected from ``signal``.
    Pressure heads are available through ``heads - elevations[None, :]``.
    """
    inp_path = Path(inp_path or DEFAULT_INP).resolve()
    if not inp_path.is_file():
        raise FileNotFoundError(inp_path)
    xi = float(xi)
    if not 0.0 < xi < 1.0:
        raise ValueError("xi must be strictly inside the pipe, between 0 and 1")
    if min(dt, rise_time, duration) <= 0 or coefficient < 0:
        raise ValueError("Time steps, speeds, duration and rise time must be positive")
    if not 0 <= baseline_end < start_time < duration:
        raise ValueError("Require 0 <= baseline_end < start_time < duration")
    log = io.StringIO()
    begun = clock.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        # Keep each warning rather than printing it. The default filter avoids
        # recording thousands of identical solver warnings at successive steps.
        warnings.simplefilter("default")
        with redirect_stdout(log), redirect_stderr(log):
            import wntr
            import tsnet

            wn = wntr.network.WaterNetworkModel(str(inp_path))
            if pipe_name not in wn.pipe_name_list:
                raise ValueError(f"{pipe_name!r} is not an original pipe")
            pipe = wn.get_link(pipe_name)
            if getattr(pipe, "check_valve", False):
                raise ValueError("This tutorial does not split check-valve pipes")
            original_start = pipe.start_node_name
            original_end = pipe.end_node_name
            original_length = float(pipe.length)
            node_names = list(wn.junction_name_list)
            elevations = np.array([wn.get_node(n).elevation for n in node_names], dtype=float)
            selected = list(node_names if sensors is None else sensors)
            if not selected or len(set(selected)) != len(selected) or any(n not in node_names for n in selected):
                raise ValueError("Sensors must be a nonempty set of distinct original junction names")
            sensor_indices = np.array([node_names.index(n) for n in selected], dtype=int)
            source_node = "IAHR_BURST_POINT"
            source_segment = "IAHR_BURST_SEGMENT"
            if source_node in wn.node_name_list or source_segment in wn.link_name_list:
                raise ValueError("Reserved tutorial burst names already exist in the network")
            wn = wntr.morph.split_pipe(wn, pipe_name, source_segment, source_node,
                                       split_at_point=xi, return_copy=True)

            with _temporary_working_directory() as folder:
                filename = folder / "event.inp"
                wntr.network.write_inpfile(wn, str(filename))
                tm = tsnet.network.TransientModel(str(filename))
                # Every pipe must have at least two MOC intervals. Interior
                # insertion can create a short segment and tighten the CFL
                # limit. TSNet subsequently adjusts dt and individual speeds.
                grid = discretize(tm, duration, dt, wave_speed)
                requested_dt = grid['cfl_limited_dt']
                effective_speeds = grid['effective_speeds']
                tm = tsnet.simulation.Initializer(tm, 0.0, "DD")
                for _, physical_pipe in tm.pipes():
                    physical_pipe._roughness = DARCY_FRICTION_FACTOR
                    physical_pipe.roughness_height = 0.
                reference = copy.deepcopy(tm) if background_subtract else None
                tm.add_burst(source_node, float(start_time), float(rise_time), float(coefficient))
                tm = tsnet.simulation.MOCSimulator(tm, "no", friction=friction)
                timestamps, heads = _extract_heads(tm, node_names)
                discharge = np.asarray(tm.get_node(source_node).emitter_discharge,
                                       dtype=float).ravel()
                if background_subtract:
                    reference = tsnet.simulation.MOCSimulator(reference, "no", friction=friction)
                    reference_time, background = _extract_heads(reference, node_names)
                    if not np.array_equal(timestamps, reference_time):
                        raise RuntimeError("Burst and background timestamps do not agree")
                    signal = heads - background
                else:
                    background = np.broadcast_to(np.median(heads[timestamps <= baseline_end], axis=0),
                                                 heads.shape).copy()
                    signal = heads - background
                actual_dt = float(tm.time_step)

    arrivals, picker = pick_arrivals(timestamps, signal, baseline_end=baseline_end,
                                    min_time=baseline_end, return_details=True)
    pressure = heads - elevations[None, :]
    warning_messages = list(dict.fromkeys(str(w.message).replace(str(inp_path), 'networks/HKUST.inp')
        .replace(str(filename), '<temporary simulation>/event.inp') for w in captured))
    minimum_pressure = float(pressure.min())
    if minimum_pressure < 0:
        warning_messages.append("Negative pressure head occurred. This event needs physical review; "
                                "TSNet's emitter backflow rule is not a cavitation model.")
    metadata = {
        "solver": "TSNet MOCSimulator", "friction": friction,
        "versions": {n: importlib.metadata.version(n) for n in ("tsnet", "wntr", "numpy")},
        "background_subtracted_for_arrival_picking": bool(background_subtract),
        "background_note": "Matched no-burst TSNet run on the same split network removes numerical "
                           "initialisation drift. This is a controlled synthetic experiment.",
        "requested_dt_s": float(dt), "cfl_limited_requested_dt_s": float(requested_dt),
        "actual_dt_s": actual_dt, "nominal_wave_speed_m_s": wave_speed,
        "wave_speed_model": 'K9 ductile-iron diameter-dependent speeds' if wave_speed is None else 'uniform override',
        "nominal_speed_min_m_s": float(grid['nominal_speeds'].min()),
        "nominal_speed_max_m_s": float(grid['nominal_speeds'].max()),
        "effective_speed_min_m_s": float(effective_speeds.min()),
        "effective_speed_max_m_s": float(effective_speeds.max()),
        "maximum_relative_speed_adjustment": float(np.max(np.abs(effective_speeds / grid['nominal_speeds'] - 1))),
        "supply_head_m": 40., "darcy_friction_factor": DARCY_FRICTION_FACTOR,
        "initial_condition": "Zero background demand and elevation; network initially at rest.",
        "minimum_pressure_head_m": minimum_pressure,
        "start_time_s": float(start_time), "rise_time_s": float(rise_time),
        "final_burst_coefficient_m3_s_sqrt_m": float(coefficient),
        "runtime_s": float(clock.monotonic() - begun),
        "warnings": warning_messages,
        "source_node_is_excluded_from_recorded_features": True,
        "observation_lead_in_s": float(baseline_end),
        "original_pipe_start": original_start, "original_pipe_end": original_end,
    }
    return {"time": timestamps, "heads": heads, "background_heads": background,
            "signal": signal, "node_names": np.asarray(node_names), "elevations": elevations,
            "sensor_names": np.asarray(selected), "sensor_indices": sensor_indices,
            "sensor_heads": heads[:, sensor_indices], "sensor_signal": signal[:, sensor_indices],
            "arrivals": arrivals, "sensor_arrivals": arrivals[sensor_indices],
            "arrival_details": picker, "burst_discharge": discharge,
            "source": {"pipe_name": str(pipe_name), "xi": xi,
                       "distance_from_start_m": xi * original_length,
                       "original_pipe_length_m": original_length},
            "actual_dt": actual_dt, "metadata": metadata}


def _resolve_source(source, net):
    if isinstance(source, dict):
        name = source.get("pipe_name", source.get("pipe"))
        xi = source.get("xi", source.get("fraction"))
    else:
        name, xi = source[:2]
    if isinstance(name, (int, np.integer)):
        names = getattr(net, "pipe_names", None)
        if names is None and isinstance(net, dict):
            names = net.get("pipe_names")
        if names is None:
            raise ValueError("Integer pipe indices need net.pipe_names")
        name = names[int(name)]
    return str(name), float(xi)


def generate_tsnet_dataset(net, test_sources, output_path, max_events=None,
                           sensors=None, output_dt=0.01, progress=True, **simulation_options):
    """Simulate held-out sources and save a reusable, pickle-free NPZ cache.

    test_sources is a sequence of (pipe_name, xi) pairs or dictionaries with
    those keys. Integer pipe indices are accepted when net.pipe_names exists.
    All junction records are saved. Traces are interpolated to a common output
    grid for compact storage; arrival times are detected on the original MOC
    grid before interpolation. A failed event raises instead of being silently
    dropped from the test set.
    """
    sources = list(test_sources)
    if max_events is not None:
        sources = sources[:int(max_events)]
    if not sources:
        raise ValueError("At least one test source is required")
    inp_path = simulation_options.pop("inp_path", None)
    if inp_path is None:
        if isinstance(net, (str, Path)):
            inp_path = net
        elif isinstance(net, dict):
            inp_path = net.get("inp_path", DEFAULT_INP)
        else:
            inp_path = getattr(net, "inp_path", DEFAULT_INP)
    simulation_options.setdefault('wave_speed', getattr(net, 'wave_speed', None))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, source in enumerate(sources):
        name, xi = _resolve_source(source, net)
        if progress:
            print(f"TSNet event {i + 1}/{len(sources)}: {name}, position {xi:.3f}", flush=True)
        result = simulate_burst(name, xi, sensors=sensors, inp_path=inp_path, **simulation_options)
        records.append(result)
        if progress:
            print(f"  {result['metadata']['runtime_s']:.1f} s; "
                  f"dt={result['actual_dt']:.5f} s; "
                  f"arrivals detected at {np.isfinite(result['arrivals']).sum()}/"
                  f"{len(result['arrivals'])} junctions", flush=True)
    common_end = min(r["time"][-1] for r in records)
    time_grid = np.arange(0.0, common_end + 0.5 * output_dt, output_dt)
    time_grid = time_grid[time_grid <= common_end]
    def on_common_grid(key):
        return np.stack([np.column_stack([
            np.interp(time_grid, r["time"], r[key][:, j])
            for j in range(len(r["node_names"]))]) for r in records]).astype(np.float32)
    package = {
        "time": time_grid, "heads": on_common_grid("heads"),
        "signal": on_common_grid("signal"), "node_names": records[0]["node_names"],
        "elevations": records[0]["elevations"],
        "arrivals": np.stack([r["arrivals"] for r in records]),
        "pipe_names": np.asarray([r["source"]["pipe_name"] for r in records]),
        "xi": np.asarray([r["source"]["xi"] for r in records]),
        "actual_dt": np.asarray([r["actual_dt"] for r in records]),
        "metadata_json": np.asarray(json.dumps([r["metadata"] for r in records], indent=2)),
        "source_json": np.asarray(json.dumps([r["source"] for r in records], indent=2)),
    }
    np.savez_compressed(output_path, **package)
    return package
