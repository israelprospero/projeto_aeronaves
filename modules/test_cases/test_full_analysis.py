import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np

airplane = dt.standard_airplane('fokker100')

dt.analyze(airplane, print_log = True, plot = True)
