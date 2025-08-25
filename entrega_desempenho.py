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

sweep_wing_v = np.arange(15,27.5,2.5)*np.pi/180
sweep_w = airplane['sweep_w']
sweep_wing_v = np.append(sweep_wing_v, sweep_w)

flap_type_v = ['plain','single slotted','double slotted', 'triple slotted']
slat_type_v = [None, 'leading edge flap', 'Kruger flaps','slats']
#print(flap_type_v)
#print(slat_type_v)
#print(sweep_wing_v)

m.plot_W0_x_Sw(airplane, Swvec, sweep_wing_v)

# m.plot_W0_x_Sw(airplane, Swvec, sweep_wing_v, flap_type_v, slat_type_v)
