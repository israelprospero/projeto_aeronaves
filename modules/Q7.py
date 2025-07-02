#Question 7 - Drag breakdown at landing condition

import designTool as dt
import utils as m
import numpy as np
import math
import pprint
import matplotlib.pyplot as plt

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi
a = 331.3  # m/s^2

# ------------------------------------------------------------ #
# ------------------- AIRPLANE 1 ----------------------------- #
# ------------------------------------------------------------ #
print('================ AIRPLANE 1 ===========================')

airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)

MLW_1 = 0.85*airplane_1['W0_guess'] #MLW - maximum landing weight - 85% of MTOW
M_landing_01 = 0.3 #first guess for landing's Mach
H_landing = 0 #at sea level
rho_landing = dt.atmosphere(H_landing)[2] #air density at sea level
V_landing_01 = a*M_landing_01 #landing's velocity based on Mach's first assumption

_, CLmax_landing_01, _ = dt.aerodynamics(airplane_1, M_landing_01, H_landing, 1, MLW_1,
                 n_engines_failed=0, highlift_config='landing',
                 lg_down=1, h_ground=0, method=2,
                 ind_drag_method='Nita',
                 ind_drag_flap_method='Roskam') #landing condition CL_max for Mach's first assumption

CL_landing_1 = CLmax_landing_01/(1.3**2)
V_stall_1 = math.sqrt(2*MLW_1/(rho_landing*CLmax_landing_01*airplane_1['S_w']))
M_stall_1 = V_stall_1/a
M_landing_1 = M_stall_1*1.3

CD_landing_1, CLmax_landing_1, dragDict_landing_1 = dt.aerodynamics(airplane_1, M_landing_1, H_landing, CL_landing_1, MLW_1,
                 n_engines_failed=0, highlift_config='landing',
                 lg_down=1, h_ground=0, method=2,
                 ind_drag_method='Nita',
                 ind_drag_flap_method='Roskam')

print(CLmax_landing_01)
print(V_stall_1)
print(CL_landing_1)
print(M_landing_1)
print(CD_landing_1)
print(CLmax_landing_1)
print(dragDict_landing_1)

# ------------------------------------------------------------ #
# ------------------- AIRPLANE 2 ----------------------------- #
# ------------------------------------------------------------ #
print('================ AIRPLANE 2 ===========================')

airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)

MLW_2 = 0.85*airplane_2['W0_guess'] #MLW - maximum landing weight
M_landing_02 = 0.3
H_landing = 0
rho_landing = dt.atmosphere(H_landing)[2]
V_landing_02 = a*M_landing_02

_, CLmax_landing_02, _ = dt.aerodynamics(airplane_2, M_landing_02, H_landing, 1, MLW_2,
                 n_engines_failed=0, highlift_config='landing',
                 lg_down=1, h_ground=0, method=2,
                 ind_drag_method='Nita',
                 ind_drag_flap_method='Roskam')

CL_landing_2 = CLmax_landing_02/(1.3**2)
V_stall_2 = math.sqrt(2*MLW_2/(rho_landing*CLmax_landing_02*airplane_2['S_w']))
M_stall_2 = V_stall_2/a
M_landing_2 = M_stall_2*1.3

CD_landing_2, CLmax_landing_2, dragDict_landing_2 = dt.aerodynamics(airplane_2, M_landing_2, H_landing, CL_landing_2, MLW_2,
                 n_engines_failed=0, highlift_config='landing',
                 lg_down=1, h_ground=0, method=2,
                 ind_drag_method='Nita',
                 ind_drag_flap_method='Roskam')

print(CLmax_landing_02)
print(V_stall_2)
print(CL_landing_2)
print(M_landing_2)
print(CD_landing_2)
print(CLmax_landing_2)
print(dragDict_landing_2)