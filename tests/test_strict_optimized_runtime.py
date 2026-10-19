import os
import subprocess
import sys
import unittest


class StrictOptimizedRuntimeTests(unittest.TestCase):
    def test_optimized_interpreter_rejects_auditor(self):
        script="from experiments.strict_engine.verify import verify_metric_record; verify_metric_record(dict(ranking=[0],truth=0,top1=0,top3=1,rr=1.))"
        r=subprocess.run([sys.executable,'-O','-c',script],text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('optimized',r.stderr.lower())

    def test_pythonoptimize_environment_rejects_auditor(self):
        env=dict(os.environ,PYTHONOPTIMIZE='1')
        r=subprocess.run([sys.executable,'-c','import experiments.strict_engine.verify'],env=env,text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('optimized',r.stderr.lower())


if __name__=='__main__':unittest.main()
