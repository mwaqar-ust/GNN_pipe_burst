"""Physical graph, neural model, portability and split invariants."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graph_tools import (load_network,make_source_split,travel_times,relative_times,
                          add_timing_noise,network_error,ambiguity_report)
from model import GraphSourceGNN,predict,save_model,load_model,_resolve_device,train_model
from run_experiments import load_test_set


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.net=load_network(min_source_pipe_length=100.)
        cls.splits=make_source_split(cls.net)

    def test_original_graph_and_source_domain(self):
        n=self.net
        self.assertEqual((n.N,n.P,n.S,n.source_pipe_count),(31,33,10,21))
        self.assertEqual(n.adjacency.shape,(31,31))
        np.testing.assert_array_equal(n.adjacency,n.adjacency.T)
        self.assertTrue(np.all(n.lengths[n.source_pipe_indices]>=100.))
        self.assertTrue(np.any(n.lengths<100.))
        # Fixed-head nodes do not transmit shortest wave paths through storage.
        self.assertTrue(np.isinf(n.travel_distances[n.boundary_indices]).all())

    def test_source_splits_and_hydraulic_events_are_disjoint(self):
        self.assertEqual([len(self.splits[k]['pipe']) for k in ('train','val','test')],[1008,252,42])
        keys=[set(zip(self.splits[k]['pipe'],self.splits[k]['xi'])) for k in ('train','val','test')]
        self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
        cached=load_test_set(self.net,self.splits)
        self.assertEqual(len(cached['pipe']),21)
        changed={k:{field:value.copy() for field,value in part.items()} for k,part in self.splits.items()}
        changed['test']['xi']+=1e-5
        with self.assertRaisesRegex(ValueError,'held-out'):
            load_test_set(self.net,changed)

    def test_cache_network_identity_accepts_windows_newlines(self):
        n=copy.deepcopy(self.net)
        source=Path(n.inp_path).read_text(encoding='utf-8-sig')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'network.inp'
            path.write_bytes(source.replace('\n','\r\n').encode('utf-8-sig'))
            n.inp_path=str(path)
            self.assertEqual(len(load_test_set(n,self.splits)['pipe']),21)
            path.write_text(source+'\n; edited network file\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'different INP'):
                load_test_set(n,self.splits)
        n=copy.deepcopy(self.net);n.nominal_pipe_wave_speeds=n.nominal_pipe_wave_speeds+1
        with self.assertRaisesRegex(ValueError,'different nominal wave speed'):
            load_test_set(n,self.splits)

    def test_material_wave_speeds_and_coordinate_projection(self):
        n=self.net
        self.assertIsNone(n.wave_speed)
        self.assertEqual(len(np.unique(n.nominal_pipe_wave_speeds)),4)
        self.assertGreater(n.nominal_pipe_wave_speeds.min(),1100.)
        self.assertLess(n.nominal_pipe_wave_speeds.max(),1300.)
        self.assertTrue(np.all(np.ptp(n.positions,axis=0)>500.))
        uniform=load_network(wave_speed=1200.,calibrated=False,min_source_pipe_length=100.)
        np.testing.assert_array_equal(uniform.nominal_pipe_wave_speeds,np.full(33,1200.))
        np.testing.assert_array_equal(uniform.pipe_wave_speeds,np.full(33,1200.))
        np.testing.assert_allclose(n.lengths,uniform.lengths)

    def test_source_travel_time_formula_and_onset_invariance(self):
        n=self.net;p=int(n.source_pipe_indices[0]);x=.31;u,v=n.ends[p]
        expected=np.minimum(x*n.lengths[p]/n.pipe_wave_speeds[p]+n.travel_distances[u],
                     (1-x)*n.lengths[p]/n.pipe_wave_speeds[p]+n.travel_distances[v])
        np.testing.assert_allclose(travel_times(n,[p],[x])[0],expected,atol=1e-12)
        t=travel_times(n,[p],[x])
        np.testing.assert_allclose(relative_times(t),relative_times(t+12.3),atol=1e-12)

    def test_timing_noise_is_reproducible_and_sensor_specific(self):
        t=np.ones((100,self.net.S))
        a=add_timing_noise(t,.010,.010,np.random.default_rng(7))
        b=add_timing_noise(t,.010,.010,np.random.default_rng(7))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a[:,0],a[:,1]))
        np.testing.assert_allclose(a/.010,np.round(a/.010),atol=1e-12)
        np.testing.assert_array_equal(relative_times(a).min(axis=1),np.zeros(100))

    def test_feature_embedding_and_pipe_output_dimensions(self):
        m=GraphSourceGNN(self.net)
        x=torch.as_tensor(self.splits['val']['relative'][:3],dtype=torch.float32)
        self.assertEqual(tuple(m.input_features(x).shape),(3,31,46))
        h,_=m.encode(x)
        self.assertEqual([tuple(v.shape) for v in h],[(3,31,32)]*3)
        y=predict(m,x.numpy())
        self.assertEqual(y['probabilities'].shape,(3,33))
        np.testing.assert_allclose(y['probabilities'].sum(axis=1),1.,atol=1e-6)
        self.assertTrue(np.all(y['probabilities'][:,~self.net.source_pipe_mask]==0))
        self.assertTrue(np.all((y['positions']>0)&(y['positions']<1)))
        self.assertEqual(sum(p.numel() for p in m.parameters()),16066)

    def test_checkpoint_roundtrip_roundoff_and_real_changes(self):
        m=GraphSourceGNN(self.net)
        x=self.splits['val']['relative'][:4]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.npz';save_model(m,path)
            loaded=load_model(self.net,path,device='cpu')
            for name in ('pipe','xi','probabilities'):
                np.testing.assert_array_equal(predict(m,x)[name],predict(loaded,x)[name])
            rounded=copy.deepcopy(self.net)
            rounded.pipe_wave_speeds=np.nextafter(rounded.pipe_wave_speeds,np.inf)
            rounded.ends=rounded.ends.astype(np.int32)
            load_model(rounded,path)
            for field in ('sensors','lengths','min_source_pipe_length'):
                changed=copy.deepcopy(self.net)
                if field=='sensors':changed.sensors=changed.sensors[::-1]
                elif field=='lengths':changed.lengths[0]+=1
                else:changed.min_source_pipe_length=101.
                with self.subTest(field=field),self.assertRaises(ValueError):load_model(changed,path)

    def test_checkpoint_contract_does_not_follow_mutated_network_arrays(self):
        net=copy.deepcopy(self.net);model=GraphSourceGNN(net)
        original=model.network_contract['lengths'].copy()
        net.lengths[0]+=1
        np.testing.assert_array_equal(model.network_contract['lengths'],original)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz';save_model(model,path)
            with self.assertRaisesRegex(ValueError,'lengths'):load_model(net,path)

    def test_cpu_fallback_and_failed_cuda_probe(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(_resolve_device('auto').type,'cpu')
            with self.assertWarns(RuntimeWarning):self.assertEqual(_resolve_device('cuda').type,'cpu')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.ones',side_effect=RuntimeError('unsupported kernel')):
            with self.assertWarns(RuntimeWarning):self.assertEqual(_resolve_device('auto').type,'cpu')

    def test_distance_metric_and_ambiguity_remain_visible(self):
        n=self.net;p=int(n.source_pipe_indices[0])
        self.assertEqual(float(network_error(n,p,.3,p,.3)),0.)
        self.assertAlmostEqual(float(network_error(n,p,.3,p,.3001)),.0001*n.lengths[p],places=7)
        q=int(n.source_pipe_indices[1])
        self.assertGreater(float(network_error(n,p,.3,q,.6)),0.)
        self.assertTrue(ambiguity_report(n)['constant_signature_pipes'])

    def test_training_rejects_overlap_and_misaligned_sensor_vectors(self):
        train={k:v[:16] for k,v in self.splits['train'].items()}
        validation={k:v[:16] for k,v in self.splits['val'].items()}
        with self.assertRaisesRegex(ValueError,'disjoint'):
            train_model(self.net,train,train,epochs=1,verbose=False,checkpoint_path=None)
        malformed=dict(validation);malformed['times']=malformed['times'][:,:-1]
        with self.assertRaisesRegex(ValueError,'sensor order'):
            train_model(self.net,train,malformed,epochs=1,verbose=False,checkpoint_path=None)

    def test_small_training_reaches_finite_loss_and_saves_best(self):
        parts={key:{k:v[:16] for k,v in self.splits[key].items()} for key in ('train','val')}
        m,h=train_model(self.net,parts['train'],parts['val'],epochs=1,threads=2,
                         checkpoint_path=None,verbose=False)
        self.assertEqual(h['best_epoch'],1)
        self.assertTrue(np.isfinite(h['val_loss']).all())
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))


if __name__=='__main__':unittest.main()
