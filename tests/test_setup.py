"""Installer policy checks without installing packages or touching drivers."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from install_torch import (detect_nvidia, install_command, parse_gpu_rows,
                           probe_torch, select_build, version_tuple)


def hardware(name='NVIDIA GeForce RTX 3070', driver='610.60', capability=8.6,
             supported=(13, 3)):
    return {'gpus':[{'name':name, 'driver':driver, 'compute_capability':capability}],
            'cuda_supported':supported}


class SetupTests(unittest.TestCase):
    def test_windows_rtx3070_and_4060_choose_verified_build(self):
        for name, capability in [('RTX 3070',8.6),('RTX 4060',8.9)]:
            with self.subTest(gpu=name):
                build,_ = select_build('Windows','AMD64',hardware(name=name,capability=capability))
                self.assertEqual(build,'cu126')

    def test_older_driver_uses_118_without_requesting_newer_runtime(self):
        build,_ = select_build('Windows','AMD64',hardware(driver='552.22',supported=(12,4)))
        self.assertEqual(build,'cu118')
        build,_ = select_build('Linux','x86_64',hardware(driver='550.100',supported=(12,4)))
        self.assertEqual(build,'cu118')

    def test_thresholds_and_unsupported_driver_choose_cpu(self):
        for system,machine,driver in [('Windows','AMD64','521.99'),('Linux','x86_64','520.60.9')]:
            with self.subTest(system=system):
                build,_ = select_build(system,machine,hardware(driver=driver,supported=(11,8)))
                self.assertEqual(build,'cpu')
        build,_ = select_build('Windows','AMD64',hardware(driver='561.17',supported=(12,6)))
        self.assertEqual(build,'cu126')

    def test_missing_driver_and_unknown_architecture_are_cpu(self):
        self.assertEqual(select_build('Linux','x86_64',{'gpus':[]})[0],'cpu')
        self.assertEqual(select_build('Windows','AMD64',hardware(capability=None))[0],'cpu')
        self.assertEqual(select_build('Windows','AMD64',hardware(capability=13.0))[0],'cpu')
        self.assertEqual(select_build('Darwin','arm64',hardware())[0],'cpu')

    def test_blackwell_requires_128_and_a_suitable_driver(self):
        newer = hardware(name='RTX 5090',capability=12.0,driver='610.60')
        self.assertEqual(select_build('Windows','AMD64',newer)[0],'cu128')
        self.assertEqual(select_build('Windows','AMD64',newer,requested='cu126')[0],'cpu')
        newer['gpus'][0]['driver']='570.65'
        self.assertEqual(select_build('Windows','AMD64',newer)[0],'cpu')

    def test_explicit_cpu_and_failed_cuda_request_are_safe(self):
        self.assertEqual(select_build('Windows','AMD64',hardware(),requested='cpu')[0],'cpu')
        old=hardware(driver='552.22',supported=(12,4))
        self.assertEqual(select_build('Windows','AMD64',old,requested='cu126')[0],'cpu')

    def test_nvidia_query_fallback_recognises_common_cards(self):
        failed = subprocess.CompletedProcess([],1,'','invalid query')
        query = subprocess.CompletedProcess([],0,'NVIDIA GeForce RTX 3070, 610.60\n','')
        header = subprocess.CompletedProcess([],0,'Driver Version: 610.60 CUDA Version: 13.3','')
        detection=detect_nvidia(Mock(side_effect=[failed,query,header]))
        self.assertEqual(detection['gpus'][0]['compute_capability'],8.6)
        self.assertEqual(detection['cuda_supported'],(13,3))
        self.assertEqual(detect_nvidia(Mock(side_effect=FileNotFoundError()))['gpus'],[])

    def test_rows_versions_and_commands(self):
        rows=parse_gpu_rows('NVIDIA GeForce RTX 4060, 610.60, 8.9\n')
        self.assertEqual(rows[0]['compute_capability'],8.9)
        self.assertGreater(version_tuple('560.35.05'),version_tuple('560.28.03'))
        command=install_command('cu126','Windows')
        self.assertIn('torch==2.7.1+cu126',command)
        self.assertEqual(command[-1],'https://download.pytorch.org/whl/cu126')
        self.assertNotIn('--index-url',install_command('cpu','Darwin'))

    def test_probe_handles_import_and_runtime_failures(self):
        missing=subprocess.CompletedProcess([],1,'','missing torch')
        self.assertFalse(probe_torch(Mock(return_value=missing))['cpu_ok'])
        timeout=Mock(side_effect=subprocess.TimeoutExpired('python',90))
        self.assertFalse(probe_torch(timeout)['cuda_ok'])
        cpu=subprocess.CompletedProcess([],0,'{"import_ok":true,"cpu_ok":true,"cuda_ok":false}\n','')
        self.assertTrue(probe_torch(Mock(return_value=cpu))['cpu_ok'])


if __name__=='__main__':
    unittest.main()
