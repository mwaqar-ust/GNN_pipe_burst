"""Exercise live teaching controls against the actual graph and signal cache."""
from pathlib import Path
import copy
import json
import sys
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graph_tools import load_network,make_source_split,travel_times,relative_times
from model import GraphSourceGNN
from run_experiments import load_test_set
import tutorial_ui as ui


class TeachingUIChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with warnings.catch_warnings(),capture_output():
            warnings.simplefilter('ignore')
            cls.net=load_network(min_source_pipe_length=100)
            cls.splits=make_source_split(cls.net,train_per_pipe=48,val_per_pipe=12,test_per_pipe=2,seed=2026)
            cls.data=load_test_set(cls.net,cls.splits)
        cls.model=GraphSourceGNN(cls.net).eval()
        cls.observation=relative_times(travel_times(cls.net,[int(cls.net.source_pipe_indices[0])],[.4]))

    def tearDown(self):
        plt.close('all')

    def test_live_subtabs_and_controls_keep_model_unchanged(self):
        before={name:value.clone() for name,value in self.model.state_dict().items()}
        factories=[lambda:ui.network_explorer(self.net),lambda:ui.adjacency_explorer(self.net),
                   lambda:ui.source_explorer(self.net,self.model),
                   lambda:ui.feature_explorer(self.net,self.model,self.observation),
                   lambda:ui.embedding_explorer(self.net,self.model,self.observation),
                   lambda:ui.event_explorer(self.net,self.model,self.data),
                   lambda:ui.sensor_explorer(self.net)]
        with capture_output():
            for factory in factories:
                widget=factory()
                self.assertGreaterEqual(len(widget._tutorial_tabs.children),2)
                for index in range(len(widget._tutorial_tabs.children)):
                    widget._tutorial_tabs.selected_index=index
                    widget._tutorial_render()
                controls=widget._tutorial_controls
                if 'layer' in controls:
                    controls['layer'].value=0
                    widget._tutorial_render()
                if 'noise' in controls:
                    controls['noise'].value=0
                    widget._tutorial_render()
                if 'event' in controls:
                    controls['event'].value=len(self.data['pipe'])-1
                    widget._tutorial_render()
                widget.close()
        for name,value in self.model.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertFalse(plt.get_fignums())

    def test_missing_detection_metrics_render_na(self):
        report={'results':{'TSNet signals':{'events':21,'localised_events':20,'missing_arrival_events':1,
                'GNN':{'Pipe accuracy (%)':50.,'Mean network distance error (m)':None}}}}
        frame=ui.metrics_table(report)
        self.assertTrue(np.isnan(frame.iloc[0]['Mean network distance error (m)']))
        fig=ui.plot_metrics(report)
        self.assertTrue(any(t.get_text()=='N/A' for ax in fig.axes for t in ax.texts))
        fig.canvas.draw()

    def test_burst_is_explicit_and_new_trace_is_rendered(self):
        n=self.net;data=self.data
        metadata=json.loads(str(np.asarray(data['metadata_json']).item()))[0]
        metadata=dict(metadata,runtime_s=.1,minimum_pressure_head_m=1.)
        record={'time':data['time'],'heads':data['heads'][0],'signal':data['signal'][0],
                'arrivals':data['arrivals'][0],'node_names':data['node_names'],
                'elevations':data['elevations'],'actual_dt':.00104045,'metadata':metadata}
        with patch('tsnet_tools.simulate_burst',return_value=record) as simulator,capture_output() as captured:
            widget=ui.burst_lab(n,self.model)
            simulator.assert_not_called()
            controls=widget._tutorial_controls
            controls['pipe'].value=n.pipe_names[int(data['pipe'][0])]
            controls['position'].value=float(data['xi'][0])
            controls['run'].click()
            simulator.assert_called_once()
            self.assertEqual(simulator.call_args.kwargs['duration'],3.)
            self.assertAlmostEqual(simulator.call_args.kwargs['dt'],.00104045)
            self.assertIsNone(simulator.call_args.kwargs['wave_speed'])
            self.assertAlmostEqual(simulator.call_args.kwargs['coefficient'],.001/np.sqrt(40.))
            self.assertFalse(controls['run'].disabled)
            widget.close()
        self.assertNotIn('Simulation needs attention',captured.stdout)
        self.assertIn('Completed in',captured.stdout)


if __name__=='__main__':
    unittest.main()
