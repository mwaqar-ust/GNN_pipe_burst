"""Independent checks of source separation, signal inputs and physical errors."""
from pathlib import Path
import csv
import sys
import unittest

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_tools import load_network, make_source_split, network_error
from run_experiments import load_test_set
from signal_story import observe_event, localise_observation
from tsnet_tools import pick_arrivals


class ScientificContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = load_network(min_source_pipe_length=100.)
        cls.splits = make_source_split(cls.net, train_per_pipe=48,
                                      val_per_pipe=12, test_per_pipe=2,
                                      seed=2026)
        cls.signals = load_test_set(cls.net, splits=cls.splits)

    def test_every_cached_source_is_held_out(self):
        """Hydraulic examples must not reuse any training/validation location."""
        locations = {name: set(zip(part['pipe'].tolist(), part['xi'].tolist()))
                     for name, part in self.splits.items()}
        self.assertEqual({name: len(value) for name, value in locations.items()},
                         {'train': 1008, 'val': 252, 'test': 42})
        self.assertFalse(locations['train'] & locations['val'])
        self.assertFalse(locations['train'] & locations['test'])
        self.assertFalse(locations['val'] & locations['test'])
        recorded = set(zip(self.signals['pipe'].tolist(),
                           self.signals['xi'].tolist()))
        self.assertEqual(len(recorded), 21)
        self.assertTrue(recorded <= locations['test'])
        self.assertFalse(recorded & locations['train'])
        self.assertFalse(recorded & locations['val'])
        self.assertTrue(np.all(self.net.lengths[self.signals['pipe']] >= 100.))
        self.assertFalse(any('BURST' in str(name)
                             for name in self.signals['node_names']))

    def test_observation_ignores_source_truth_and_simulated_onset(self):
        """Changing explanatory truth must never change the GNN observation."""
        altered = dict(self.signals)
        altered['pipe'] = np.zeros_like(self.signals['pipe'])
        altered['xi'] = np.full_like(self.signals['xi'], .123)
        # This would fail immediately if the observation builder read the
        # simulation metadata, including the known event onset.
        altered['metadata_json'] = 'not readable JSON'
        for event in (0, 8, 18):
            original = observe_event(self.net, self.signals, event,
                                     noise_std_m=.005, jitter_ms=10., seed=42)
            changed = observe_event(self.net, altered, event,
                                    noise_std_m=.005, jitter_ms=10., seed=42)
            for field in ('noise', 'signal_picks_s', 'arrivals_s', 'relative_s'):
                np.testing.assert_array_equal(original[field], changed[field])

    def test_clean_recordings_supply_all_arrivals_on_the_recording_grid(self):
        """The tutorial repicks the displayed 100 Hz records, not hidden data."""
        time = self.signals['time']
        np.testing.assert_allclose(np.diff(time), .010, atol=1e-12, rtol=0)
        altered = dict(self.signals)
        altered['arrivals'] = np.full_like(self.signals['arrivals'], np.nan)
        for event in range(len(self.signals['pipe'])):
            obs = observe_event(self.net, altered, event)
            self.assertTrue(obs['detected'].all())
            self.assertEqual(len(obs['arrivals_s']), self.net.S)
            np.testing.assert_allclose(obs['arrivals_s']/.010,
                                       np.round(obs['arrivals_s']/.010),
                                       atol=1e-10, rtol=0)

    def test_noise_streams_are_repeatable_and_separate(self):
        first = observe_event(self.net, self.signals, 0, .005, 0., seed=31)
        repeated = observe_event(self.net, self.signals, 0, .005, 0., seed=31)
        extra_jitter = observe_event(self.net, self.signals, 0, .005, 20., seed=31)
        np.testing.assert_array_equal(first['noise'], repeated['noise'])
        np.testing.assert_array_equal(first['arrivals_s'], repeated['arrivals_s'])
        np.testing.assert_array_equal(first['noisy_signal'], extra_jitter['noisy_signal'])
        np.testing.assert_array_equal(first['signal_picks_s'], extra_jitter['signal_picks_s'])
        self.assertTrue(np.all(first['extra_jitter_s'] == 0.))
        self.assertTrue(np.any(extra_jitter['extra_jitter_s'] != 0.))

    def test_zero_noise_controls_reproduce_the_signal_only_picker(self):
        obs = observe_event(self.net, self.signals, 0, 0., 0., seed=31)
        expected = pick_arrivals(obs['time'], obs['clean_signal'],
                                 baseline_end=.3, min_time=.3)
        np.testing.assert_array_equal(obs['signal_picks_s'], expected)
        np.testing.assert_array_equal(obs['noise'], np.zeros_like(obs['noise']))
        np.testing.assert_allclose(obs['arrivals_s'], np.floor(expected/.010+.5)*.010,
                                   rtol=0, atol=1e-12)

    def test_missing_signal_arrival_blocks_inference_without_imputation(self):
        altered = dict(self.signals)
        altered['signal'] = self.signals['signal'].copy()
        column = list(self.signals['node_names']).index(self.net.sensors[0])
        altered['signal'][0, :, column] = 0.
        obs = observe_event(self.net, altered, 0)
        self.assertFalse(obs['detected'][0])
        self.assertTrue(np.isnan(obs['arrivals_s'][0]))
        self.assertTrue(np.isnan(obs['relative_s']).all())
        # None cannot perform inference, so this also checks that the guard
        # returns before trying to use a model or substitute an arrival.
        result = localise_observation(self.net, None, obs)
        self.assertFalse(result['available'])
        self.assertIn(self.net.sensors[0], result['message'])

    def test_hydraulic_graph_preserves_supplied_hkust_pipe_geometry(self):
        """Compare the independently supplied CSV with the actual INP graph."""
        with (ROOT/'networks/hkust_primary_network.csv').open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), self.net.P)
        self.assertEqual((self.net.N, self.net.P), (31, 33))
        for row in rows:
            index = self.net.pipe_names.index(row['pipe_id'])
            u, v = self.net.ends[index]
            self.assertEqual(self.net.names[u], row['start_node_id'])
            self.assertEqual(self.net.names[v], row['end_node_id'])
            self.assertAlmostEqual(self.net.lengths[index], float(row['length']), places=8)
            self.assertAlmostEqual(self.net.diameters[index]*1000., float(row['diameter']), places=8)
        self.assertEqual([self.net.names[i] for i in self.net.boundary_indices], ['T1'])
        self.assertGreater(np.ptp(self.net.nominal_pipe_wave_speeds), 50.)
        # Longitude/latitude must be projected before coordinate features.
        self.assertGreater(np.ptp(self.net.positions[:, 0]), 500.)

    def test_network_error_matches_independently_split_physical_graph(self):
        """Insert both points as nodes and solve an independent shortest path."""
        net = self.net
        rng = np.random.default_rng(1234)
        for event in range(60):
            true_pipe, predicted_pipe = map(int, rng.choice(net.P, 2))
            true_xi, predicted_xi = rng.uniform(.01, .99, 2)
            if event % 4 == 0:
                predicted_pipe = true_pipe
            graph = nx.MultiGraph()
            graph.add_nodes_from(range(net.N))
            for pipe, (u, v) in enumerate(net.ends):
                cuts = [(0., int(u)), (1., int(v))]
                if pipe == true_pipe:
                    cuts.append((true_xi, 'true'))
                if pipe == predicted_pipe:
                    cuts.append((predicted_xi, 'predicted'))
                cuts.sort(key=lambda pair: pair[0])
                for (left, a), (right, b) in zip(cuts, cuts[1:]):
                    graph.add_edge(a, b, weight=(right-left)*net.lengths[pipe])
            for (u, v), kind in zip(net.link_ends, net.link_types):
                if kind != 'Pipe':
                    graph.add_edge(int(u), int(v), weight=0.)
            expected = nx.shortest_path_length(graph, 'true', 'predicted',
                                               weight='weight')
            actual = float(network_error(net, true_pipe, true_xi,
                                         predicted_pipe, predicted_xi))
            self.assertAlmostEqual(actual, expected, places=7)


if __name__ == '__main__':
    unittest.main()
