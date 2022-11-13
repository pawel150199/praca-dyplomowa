import unittest
import warnings
import os
import warnings
from execution.analysis.ImbalanceRatio import IR


"""Test e2e ImbalanceRatio class"""


class Test(unittest.TestCase):
    def teste2e(self):
        obj = IR()
        obj.calculate()
        obj.tab(True)
        os.chdir('../latexTable')
        self.assertTrue(os.path.exists('IR.txt'))

if __name__=='__main__':
    # Ignore warnings
    warnings.filterwarnings("ignore")
    unittest.main()
