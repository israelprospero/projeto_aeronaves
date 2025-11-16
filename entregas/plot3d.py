import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m

airplane = dt. standard_airplane ('fokker100')
dt.geometry(airplane)
dt.plot3d(airplane)