"""Graph travel-time prior, source examples and observability for HKUST.

All pipe labels are zero-based indices into ``net.pipe_names``.  A position
``xi`` is measured from the original INP pipe's start node towards its end node.
The original network is never split here. Temporary source junctions belong
only to the independent TSNet simulator. By default, the prior uses effective
pipe wave speeds from one source-independent TSNet discretisation of the
original network. It never reads source-specific simulations or burst signals.
"""

from dataclasses import dataclass
from pathlib import Path
from contextlib import redirect_stdout
import io
import tempfile
import warnings

import networkx as nx
import numpy as np
import wntr

from network_config import nominal_pipe_speeds, discretize


# The archived HKUST ten-junction layout, retained as an interpretable
# teaching configuration. It was not selected using this tutorial's test data.
DEFAULT_SENSORS = (
    "J10", "J84", "J102", "J107", "J88", "J62", "J79", "J46", "J16", "J51",
)


@dataclass
class NetworkData:
    names: tuple
    pipe_names: tuple
    ends: np.ndarray
    lengths: np.ndarray
    diameters: np.ndarray
    positions: np.ndarray
    adjacency: np.ndarray
    sensors: tuple
    sensor_indices: np.ndarray
    travel_distances: np.ndarray
    node_distances: np.ndarray
    wave_speed: float | None
    nominal_pipe_wave_speeds: np.ndarray
    pipe_wave_speeds: np.ndarray
    reference_dt: float
    link_names: tuple
    link_ends: np.ndarray
    link_types: tuple
    link_status: np.ndarray
    boundary_indices: np.ndarray
    propagation_distances: np.ndarray
    graph: nx.Graph
    propagation_graph: nx.Graph
    inp_path: str
    metadata: dict
    min_source_pipe_length: float = 0.0

    @property
    def N(self):
        return len(self.names)

    @property
    def P(self):
        return len(self.pipe_names)

    @property
    def S(self):
        return len(self.sensors)

    @property
    def source_pipe_mask(self):
        """Eligible source pipes; every physical pipe remains in the graph."""
        threshold = float(self.min_source_pipe_length)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Minimum source pipe length must be finite and nonnegative [m].")
        mask = np.asarray(self.lengths) >= threshold
        if not mask.any():
            raise ValueError(f"No source pipes have length >= {threshold:g} m. Lower the threshold.")
        return mask

    @property
    def source_pipe_indices(self):
        """Original pipe indices whose lengths meet the inclusive threshold."""
        return np.flatnonzero(self.source_pipe_mask)

    @property
    def source_pipe_count(self):
        return int(np.count_nonzero(self.source_pipe_mask))

    @property
    def sensor_mask(self):
        mask = np.zeros(self.N, dtype=float)
        mask[self.sensor_indices] = 1.0
        return mask


def _distance_matrix(graph, n):
    result = np.full((n, n), np.inf)
    for node, distances in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
        for other, distance in distances.items():
            result[node, other] = distance
    return result


def _add_shortest_edge(graph, u, v, cost):
    """Retain the cheaper link if the network has parallel links."""
    if not graph.has_edge(u, v) or cost < graph[u][v]["weight"]:
        graph.add_edge(u, v, weight=float(cost))


def _choose_sensors(distances, junctions, count=10):
    """Deterministic farthest-point placement; a starting layout, not an optimum."""
    junctions = np.asarray(junctions, dtype=int)
    if len(junctions) < count:
        raise ValueError("Sensor count exceeds the number of eligible junctions.")
    local = distances[np.ix_(junctions, junctions)]
    if not np.all(np.isfinite(local)):
        raise ValueError("Eligible junctions are not in one propagation component.")
    # Begin at the travel-time medoid, then spread sensors over the graph.
    selected = [int(np.argmin(local.mean(axis=1)))]
    nearest = local[:, selected[0]].copy()
    while len(selected) < count:
        nearest[selected] = -np.inf
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        nearest = np.minimum(nearest, local[:, chosen])
    return junctions[selected]


def load_network(sensors=None, wave_speed=None, inp_path=None, sensor_count=10,
                 prior_dt=0.00104045, calibrated=True, min_source_pipe_length=0.0,
                 geographic_coordinates=True):
    """Read HKUST and create geometry, propagation distances and sensor features.

    The t=0 EPANET result supplies the initial open/closed link status. Open
    pumps and valves are approximated as zero-time interfaces in the first-
    arrival prior. Pressure-wave paths cannot transit ordinary reservoir or
    tank nodes, which TSNet treats as fixed-head transient boundaries. The
    distance-error metric is a separate geometric shortest-path calculation.

    TSNet adjusts each pipe's wave speed to place characteristics on its space-
    time grid. With calibrated=True, this discretisation-aware prior reads the
    resulting effective speeds from the ORIGINAL unsplit network, using the
    requested prior_dt. This is performed before any source is specified.
    Splitting a pipe may still cause small event-specific numerical differences;
    those are not fed back into the prior. Set calibrated=False to compare with
    a purely nominal-speed prior. ``net.wave_speed=None`` uses the archived
    diameter-dependent ductile-iron material model; a positive scalar requests
    one uniform nominal speed. ``net.nominal_pipe_wave_speeds`` records the
    resulting per-pipe nominal values in original pipe order.

    The bundled INP stores longitude/latitude. With geographic_coordinates=True,
    these are projected to local metres for plotting and structural features.
    Physical path costs always use stored pipe lengths. For a custom INP whose
    coordinates are already Cartesian, set geographic_coordinates=False.

    ``min_source_pipe_length`` [m] restricts source locations to original pipes at least this long. It never removes
    pipes from the hydraulic model, propagation graph or distance metric.
    """
    if inp_path is None:
        inp_path = Path(__file__).parent / "networks" / "HKUST.inp"
    inp_path = Path(inp_path).resolve()
    if wave_speed is not None and (not np.isfinite(wave_speed) or wave_speed <= 0):
        raise ValueError("Wave speed must be positive and finite, or None for the material model.")
    if prior_dt <= 0:
        raise ValueError("Prior discretisation time step must be positive.")
    min_source_pipe_length = float(min_source_pipe_length)
    if not np.isfinite(min_source_pipe_length) or min_source_pipe_length < 0:
        raise ValueError("Minimum source pipe length must be finite and nonnegative [m].")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Not all curves were used.*")
        warnings.filterwarnings("ignore", message="Changing the headloss formula.*")
        wn = wntr.network.WaterNetworkModel(str(inp_path))
    wn.options.time.duration = 0
    with tempfile.TemporaryDirectory(prefix="iahr_initial_state_") as tmp:
        result = wntr.sim.EpanetSimulator(wn).run_sim(
            file_prefix=str(Path(tmp) / "initial")
        )
    status = result.link["status"].iloc[0]
    names = tuple(wn.node_name_list)
    lookup = {name: index for index, name in enumerate(names)}
    pipes = list(wn.pipes())
    links = list(wn.links())
    pipe_names = tuple(name for name, _ in pipes)
    nominal_speeds = nominal_pipe_speeds(wn, wave_speed)
    if calibrated:
        import tsnet
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore", message="Not all curves were used.*")
            warnings.filterwarnings("ignore", message="Changing the headloss formula.*")
            reference = tsnet.network.TransientModel(str(inp_path))
            discretize(reference, duration=3.0, dt=float(prior_dt), wave_speed=wave_speed)
        pipe_wave_speeds = np.array([reference.get_link(name).wavev for name in pipe_names], dtype=float)
        reference_dt = float(reference.time_step)
    else:
        pipe_wave_speeds = nominal_speeds.copy()
        reference_dt = float(prior_dt)
    speed_by_pipe = dict(zip(pipe_names, pipe_wave_speeds))
    ends = np.array([[lookup[p.start_node_name], lookup[p.end_node_name]] for _, p in pipes])
    lengths = np.array([p.length for _, p in pipes], dtype=float)
    if not np.any(lengths >= min_source_pipe_length):
        raise ValueError(f"No source pipes have length >= {min_source_pipe_length:g} m. Lower the threshold.")
    diameters = np.array([p.diameter for _, p in pipes], dtype=float)
    positions = np.array([wn.get_node(name).coordinates for name in names], dtype=float)
    if geographic_coordinates:
        longitude, latitude = positions.T
        if np.any(np.abs(longitude) > 180) or np.any(np.abs(latitude) > 90):
            raise ValueError("Expected longitude/latitude coordinates; use geographic_coordinates=False for Cartesian coordinates.")
        earth_radius = 6371000.0
        positions = earth_radius * np.column_stack((
            np.deg2rad(longitude-longitude.mean()) * np.cos(np.deg2rad(latitude.mean())),
            np.deg2rad(latitude-latitude.mean())))
    boundary_names = set(wn.reservoir_name_list + wn.tank_name_list)
    boundary = np.array([lookup[name] for name in names if name in boundary_names], dtype=int)
    junctions = np.array([lookup[name] for name in wn.junction_name_list], dtype=int)
    n = len(names)
    geometry = nx.Graph()
    geometry.add_nodes_from(range(n))
    propagation = nx.Graph()
    propagation.add_nodes_from(junctions.tolist())
    adjacency = np.zeros((n, n), dtype=float)
    link_ends, link_types, link_status = [], [], []
    for name, link in links:
        u, v = lookup[link.start_node_name], lookup[link.end_node_name]
        is_pipe = link.link_type == "Pipe"
        length = float(link.length) if is_pipe else 0.0
        opened = bool(status[name] > 0)
        link_ends.append((u, v))
        link_types.append(link.link_type)
        link_status.append(int(opened))
        # The metric follows the original physical network, including storage
        # endpoints; it does not pretend to describe wave transmission there.
        _add_shortest_edge(geometry, u, v, length)
        if opened:
            adjacency[u, v] = adjacency[v, u] = 1.0
            if u not in boundary and v not in boundary:
                travel_cost = length / speed_by_pipe[name] if is_pipe else 0.0
                _add_shortest_edge(propagation, u, v, travel_cost)
    propagation_distances = _distance_matrix(propagation, n)
    selection_note = "Explicit user-selected junction list."
    if sensors is None:
        if sensor_count == 10 and set(DEFAULT_SENSORS).issubset(names):
            sensors = DEFAULT_SENSORS
            sensor_indices = np.array([lookup[name] for name in sensors], dtype=int)
            selection_note = (
                "Archived HKUST ten-junction teaching layout; retained before generating "
                "this tutorial's training, validation or test examples; not claimed optimal."
            )
        else:
            sensor_indices = _choose_sensors(propagation_distances, junctions, sensor_count)
            sensors = tuple(names[i] for i in sensor_indices)
            selection_note = "Travel-time farthest-point heuristic, starting at the medoid; not claimed optimal."
    else:
        sensors = tuple(sensors)
        if len(sensors) < 2 or len(set(sensors)) != len(sensors):
            raise ValueError("Choose at least two distinct junction sensors.")
        unknown = set(sensors) - set(names)
        if unknown:
            raise ValueError(f"Unknown sensor node names: {sorted(unknown)}")
        if set(sensors) & boundary_names:
            raise ValueError("Reservoir and tank nodes are fixed-head boundaries, not pressure-arrival sensors.")
        sensor_indices = np.array([lookup[name] for name in sensors], dtype=int)
    net = NetworkData(
        names=names, pipe_names=pipe_names, ends=ends, lengths=lengths,
        diameters=diameters, positions=positions, adjacency=adjacency,
        sensors=tuple(sensors), sensor_indices=sensor_indices,
        travel_distances=propagation_distances[:, sensor_indices],
        node_distances=_distance_matrix(geometry, n),
        wave_speed=None if wave_speed is None else float(wave_speed),
        nominal_pipe_wave_speeds=nominal_speeds, pipe_wave_speeds=pipe_wave_speeds, reference_dt=reference_dt,
        link_names=tuple(name for name, _ in links), link_ends=np.array(link_ends),
        link_types=tuple(link_types), link_status=np.array(link_status),
        boundary_indices=boundary, propagation_distances=propagation_distances,
        graph=geometry, propagation_graph=propagation, inp_path=str(inp_path),
        min_source_pipe_length=min_source_pipe_length,
        metadata={
            "reservoirs": list(wn.reservoir_name_list), "tanks": list(wn.tank_name_list),
            "closed_links": [name for name, _ in links if status[name] == 0],
            "sensor_selection": selection_note,
            "prior_type": "discretisation-aware source-independent graph prior" if calibrated else "nominal-speed graph prior",
            "prior_calibrated": bool(calibrated),
            "prior_dt_requested_seconds": float(prior_dt),
            "reference_dt_seconds": reference_dt,
            "nominal_wave_speed_m_per_s": None if wave_speed is None else float(wave_speed),
            "nominal_wave_speed_min_m_per_s": float(nominal_speeds.min()),
            "nominal_wave_speed_max_m_per_s": float(nominal_speeds.max()),
            "wave_speed_model": "ductile iron K9 diameter-dependent model" if wave_speed is None else "uniform scalar override",
            "case_name": "HKUST",
            "coordinate_projection": "local equirectangular metres from INP longitude/latitude" if geographic_coordinates else "input Cartesian coordinates",
            "effective_wave_speed_min_m_per_s": float(pipe_wave_speeds.min()),
            "effective_wave_speed_max_m_per_s": float(pipe_wave_speeds.max()),
            "wave_speed_note": "Effective speeds use the original unsplit network only; no source-specific data, coordinates, split lengths, or transient records enter this calibration.",
            "prior_assumptions": "fixed-head storage blocks transit; open devices have zero delay; mean-flow advection neglected",
            "min_source_pipe_length_m": min_source_pipe_length,
            "source_pipe_count": int(np.count_nonzero(lengths >= min_source_pipe_length)),
        },
    )
    # Source restriction never deletes physical pipes. Validate only candidates.
    if any(status[pipe_names[p]] == 0 for p in net.source_pipe_indices):
        raise ValueError("The network contains closed source pipes. Define the candidate set explicitly before training.")
    travel_times(net, net.source_pipe_indices, np.full(net.source_pipe_count, 0.5))
    return net


def travel_times(net, pipe_indices, xi):
    """Earliest travel times [events, sensors] from continuous pipe sources.

    A source on pipe e=(u,v), length L, reaches sensor s by the shorter of
    xi*L/a_e + D(u,s) and (1-xi)*L/a_e + D(v,s), with the same pipe-specific
    effective speed a_e used in graph paths. An endpoint at a fixed-head boundary
    has D=inf and cannot supply a transmitted path to another junction.
    """
    pipe, fraction = np.broadcast_arrays(np.asarray(pipe_indices, dtype=int), np.asarray(xi, dtype=float))
    pipe, fraction = pipe.reshape(-1), fraction.reshape(-1)
    if np.any(pipe < 0) or np.any(pipe >= net.P):
        raise ValueError("Pipe index outside the original network.")
    if np.any(fraction <= 0) or np.any(fraction >= 1):
        raise ValueError("Use interior sources, 0 < xi < 1; junction pipe labels are ambiguous.")
    ends = net.ends[pipe]
    tau = net.lengths[pipe] / net.pipe_wave_speeds[pipe]
    via_start = fraction[:, None] * tau[:, None] + net.travel_distances[ends[:, 0]]
    via_end = (1 - fraction[:, None]) * tau[:, None] + net.travel_distances[ends[:, 1]]
    times = np.minimum(via_start, via_end)
    if not np.all(np.isfinite(times)):
        raise ValueError("At least one source cannot reach every selected sensor in the propagation graph.")
    return times


def relative_times(times):
    """Remove unknown source onset by referencing the first sensor arrival."""
    times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("Arrival times must be finite.")
    return times - times.min(axis=-1, keepdims=True)


def timing_noise_config(sigma=0.010, quantum=0.010, picking_error_samples=None,
                        sampling_hz=100.0):
    """Validate and describe the selected timestamp uncertainty model."""
    sigma, quantum, sampling_hz = float(sigma), float(quantum), float(sampling_hz)
    if not np.isfinite([sigma, quantum, sampling_hz]).all() or sigma < 0 or quantum < 0 or sampling_hz <= 0:
        raise ValueError('Use finite nonnegative sigma/quantum and a positive sampling_hz.')
    config = {'mode': 'gaussian', 'sigma_s': sigma, 'quantum_s': quantum,
              'sampling_hz': sampling_hz}
    if picking_error_samples is not None:
        bounds = np.asarray(picking_error_samples, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or np.any(bounds != np.floor(bounds))
                or bounds[0] < 0 or bounds[1] < bounds[0]):
            raise ValueError('picking_error_samples must be two integers: 0 <= minimum <= maximum.')
        if sigma != 0:
            raise ValueError('Set sigma=0 when using picking_error_samples; this selects sample shifts instead of Gaussian jitter.')
        config.update(mode='sample_picking', min_samples=int(bounds[0]), max_samples=int(bounds[1]))
    return config


def add_timing_noise(times, sigma=0.010, quantum=0.010, rng=None, *,
                     picking_error_samples=None, sampling_hz=100.0):
    """Add independent timestamp errors, then optionally quantise [s].

    Call relative_times afterwards. Adding noise before differencing retains
    the shared-reference correlation present in relative arrival measurements.
    By default errors are Gaussian. With picking_error_samples=(2, 10), sigma
    must be zero: each event/sensor gets an independent integer magnitude in
    [2, 10] and independent sign, divided by sampling_hz. Quantisation adds its
    own rounding error; quantum=0 preserves exactly the specified time shifts.
    """
    config = timing_noise_config(sigma, quantum, picking_error_samples, sampling_hz)
    rng = np.random.default_rng() if rng is None else rng
    times = np.asarray(times, dtype=float)
    if not np.isfinite(times).all():
        raise ValueError('Arrival times must be finite.')
    if config['mode'] == 'sample_picking':
        magnitude = rng.integers(config['min_samples'], config['max_samples'] + 1, size=times.shape)
        sign = rng.choice([-1, 1], size=times.shape)
        noisy = times + sign * magnitude / config['sampling_hz']
    else:
        noisy = times + rng.normal(0, config['sigma_s'], size=times.shape)
    if quantum:
        noisy = np.round(noisy / quantum) * quantum
    return noisy


def make_source_split(net, train_per_pipe=48, val_per_pipe=12, test_per_pipe=2, seed=2026):
    """Balanced, disjoint source locations; noise replicas stay in their split.

    Each eligible pipe receives a reproducible stratified set of locations.
    Random draws still visit every original pipe so changing only the length
    threshold retains exactly the same locations on pipes present in both runs.
    The 2% endpoint margin avoids an arbitrary pipe label at an exact junction.
    The model itself predicts a continuous fraction anywhere in (0,1).
    """
    requested_counts = (train_per_pipe, val_per_pipe, test_per_pipe)
    if any(not np.isfinite(x) or x != int(x) or x < 1 for x in requested_counts):
        raise ValueError("Each split needs a positive integer number of sources per pipe.")
    counts = [int(x) for x in requested_counts]
    rng = np.random.default_rng(seed)
    total = sum(counts)
    storage = {name: {"pipe": [], "xi": []} for name in ("train", "val", "test")}
    source_mask = net.source_pipe_mask
    for p in range(net.P):
        fractions = 0.02 + 0.96 * (np.arange(total) + rng.uniform(0.15, 0.85, total)) / total
        fractions = fractions[rng.permutation(total)]
        if not source_mask[p]:
            continue
        begin = 0
        for name, count in zip(storage, counts):
            storage[name]["pipe"].extend([p] * count)
            storage[name]["xi"].extend(fractions[begin:begin + count])
            begin += count
    for data in storage.values():
        data["pipe"] = np.asarray(data["pipe"], dtype=int)
        data["xi"] = np.asarray(data["xi"], dtype=float)
        data["times"] = travel_times(net, data["pipe"], data["xi"])
        data["relative"] = relative_times(data["times"])
    return storage


def signature_grid(net, points_per_pipe=201, endpoint_margin=0.001):
    """Dense travel-signature grid on eligible source pipes, with original IDs."""
    if points_per_pipe < 2 or not 0 < endpoint_margin < 0.5:
        raise ValueError("Use at least two interior candidates per pipe.")
    fractions = np.linspace(endpoint_margin, 1 - endpoint_margin, points_per_pipe)
    source_pipes = net.source_pipe_indices
    pipe = np.repeat(source_pipes, points_per_pipe)
    xi = np.tile(fractions, len(source_pipes))
    times = travel_times(net, pipe, xi)
    return {"pipe": pipe, "xi": xi, "times": times, "relative": relative_times(times)}


def network_error(net, true_pipe, true_xi, pred_pipe, pred_xi):
    """Shortest distance [m] along the original network between two pipe points.

    For the same pipe include direct along-pipe separation. Also check all four
    endpoint combinations, because a loop can provide a shorter route.
    """
    tp, tx, pp, px = np.broadcast_arrays(true_pipe, true_xi, pred_pipe, pred_xi)
    shape = tp.shape
    tp, pp = np.asarray(tp, dtype=int).ravel(), np.asarray(pp, dtype=int).ravel()
    tx, px = np.asarray(tx, dtype=float).ravel(), np.asarray(px, dtype=float).ravel()
    if np.any(tp < 0) or np.any(tp >= net.P) or np.any(pp < 0) or np.any(pp >= net.P):
        raise ValueError("Pipe index outside the network.")
    if np.any((tx < 0) | (tx > 1) | (px < 0) | (px > 1)):
        raise ValueError("Position fractions must lie in [0,1].")
    true_offsets = np.column_stack((tx, 1 - tx)) * net.lengths[tp, None]
    pred_offsets = np.column_stack((px, 1 - px)) * net.lengths[pp, None]
    error = np.where(tp == pp, np.abs(tx - px) * net.lengths[tp], np.inf)
    for i in range(2):
        for j in range(2):
            distance = true_offsets[:, i] + net.node_distances[net.ends[tp, i], net.ends[pp, j]] + pred_offsets[:, j]
            error = np.minimum(error, distance)
    return error.reshape(shape)


def ambiguity_report(net, points_per_pipe=41, tolerance=1e-8):
    """Expose intrinsic non-uniqueness of first-arrival localisation.

    This diagnostic uses an exact synthetic grid, not training or test outcomes.
    A constant signature means no relative-arrival method can infer position
    along that pipe under these assumptions. Other duplicate signatures can
    span only part of a pipe or more than one pipe.
    """
    candidates = signature_grid(net, points_per_pipe, endpoint_margin=0.02)
    source_pipes = net.source_pipe_indices
    signatures = candidates["relative"].reshape(len(source_pipes), points_per_pipe, net.S)
    spans = np.ptp(signatures, axis=1).max(axis=1)
    constant = source_pipes[spans < tolerance]
    quantised = np.round(candidates["relative"] / tolerance).astype(np.int64)
    _, inverse, counts = np.unique(quantised, axis=0, return_inverse=True, return_counts=True)
    boundary_pipes = source_pipes[np.isin(net.ends[source_pipes], net.boundary_indices).any(axis=1)]
    return {
        "constant_signature_pipes": [net.pipe_names[p] for p in constant],
        "boundary_pipes": [net.pipe_names[p] for p in boundary_pipes],
        "duplicate_signature_groups": int(np.sum(counts > 1)),
        "fraction_grid_points_with_a_duplicate": float(np.mean(counts[inverse] > 1)),
        "grid_points_per_pipe": int(points_per_pipe),
        "tolerance_seconds": float(tolerance),
        "note": "Grid diagnostic only; sensor placement is not guaranteed optimal. Ambiguous cases remain in the reported GNN metrics.",
    }


def summary_metrics(net, true_pipe, true_xi, pred_pipe, pred_xi):
    """The two tutorial metrics, with no extra performance scores."""
    return {
        "Pipe accuracy (%)": float(100 * np.mean(np.asarray(true_pipe) == np.asarray(pred_pipe))),
        "Mean network distance error (m)": float(np.mean(network_error(net, true_pipe, true_xi, pred_pipe, pred_xi))),
    }
