import unittest
import sys, os
import numpy as np
sys.path.append('../analysis')
from ImbalanceRatio import IR

"""Test e2e ImbalanceRatio class"""

class Test(unittest.TestCase):
    def teste2e(self):
        obj = IR(['glass', 'appendicitis', 'balance', 'banana', 'bupa'])
        obj.calculate()
        obj.tab(True)
        os.chdir('../latexTable')
        self.assertTrue(os.path.exists('IR.txt'))

if __name__=='__main__':
    unittest.main()
