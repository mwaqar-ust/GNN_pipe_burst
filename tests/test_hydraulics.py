"""Physical input and detector checks for the HKUST teaching case."""
import unittest
from pathlib import Path
import numpy as np
import wntr
from network_config import nominal_pipe_speeds, ductile_iron_wave_speed
from tsnet_tools import pick_arrivals

ROOT=Path(__file__).resolve().parents[1]


class HKUSTHydraulicsTests(unittest.TestCase):
    def test_material_units_and_uniform_override(self):
        wn=wntr.network.WaterNetworkModel(str(ROOT/'networks/HKUST.inp'))
        self.assertEqual((len(wn.junction_name_list),len(wn.pipe_name_list)),(30,33))
        speeds=nominal_pipe_speeds(wn)
        self.assertEqual(speeds.shape,(33,))
        self.assertAlmostEqual(float(speeds.min()),1137.405033,places=5)
        self.assertAlmostEqual(float(speeds.max()),1231.378063,places=5)
        self.assertTrue(np.array_equal(nominal_pipe_speeds(wn,1200.),np.full(33,1200.)))
        with self.assertRaises(ValueError):nominal_pipe_speeds(wn,-1)
        with self.assertRaises(ValueError):ductile_iron_wave_speed([.3])

    def test_small_numerical_departures_are_not_arrivals(self):
        t=np.arange(0,1,.01)
        y=np.zeros((len(t),2))
        y[t>=.35,0]=2e-5
        y[t>=.60,0]=-.02
        arrivals=pick_arrivals(t,y)
        self.assertGreater(arrivals[0],.58)
        self.assertLessEqual(arrivals[0],.60)
        self.assertTrue(np.isnan(arrivals[1]))

    def test_required_consecutive_crossings_reject_spike(self):
        t=np.arange(0,1,.01)
        y=np.zeros(len(t));y[40]=-.05;y[t>=.7]=-.02
        a,detail=pick_arrivals(t,y,return_details=True)
        self.assertGreater(a[0],.68)
        self.assertEqual(detail['indices'][0],70)
        self.assertAlmostEqual(detail['threshold_m'][0],.001)


if __name__=='__main__':unittest.main()
