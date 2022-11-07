import unittest
import sys, os
import warnings
sys.path.append('../analysis')
from ImbalanceRatio import IR

"""Test e2e ImbalanceRatio class"""
# Ignore warnings
warnings.filterwarnings("ignore")
class Test(unittest.TestCase):
    def teste2e(self):
        obj = IR(['glass', 'appendicitis', 'balance', 'banana', 'bupa'])
        obj.calculate()
        obj.tab(True)
        os.chdir('../latexTable')
        self.assertTrue(os.path.exists('IR.txt'))

if __name__=='__main__':
    unittest.main()
