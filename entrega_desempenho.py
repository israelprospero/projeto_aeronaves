from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m

airplane = dt. standard_airplane ('my_airplane_1')

dt.geometry(airplane)
# dt.plot3d(airplane)

Swvec = np.arange(50, 151, 1)
m.plot_T0_x_Sw(airplane, Swvec)
