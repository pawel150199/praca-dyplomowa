from distutils.log import error
import sys, os
import numpy as np

sys.path.append('../analysis')
from ImbalanceRatio import IR


def main():
    obj = IR(['glass'])
    obj.calculate()
    obj.tab(True)
    os.chdir('../latexTable')
    if not os.path.exists('IR.txt'):
        error('Nie działa!!!!')

if __name__=='__main__':
    main()
