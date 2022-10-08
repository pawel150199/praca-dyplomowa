import os
import unittest
import numpy as np
from ..analysis.ImbalanceRatio import IR

class Test(unittest.TestCase):
    def e2eTest(self):
        obj = IR(['glass'])
        obj.calculate()
        obj.tab(True)
        os.chdir('../latexTable')
        os.path.exists('IR.txt')
