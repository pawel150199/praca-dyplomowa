import sys
import unittest
import numpy as np

sys.path.append('../algorithms')
from Bagging import BaggingClassifier

class Test(unittest.TestCase):
    def e2eTest(self):
        obj = IR(['glass'])
        obj.calculate()
        obj.tab(True)
        os.chdir('../latexTable')
        os.path.exists('IR.txt')
