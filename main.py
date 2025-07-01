import modules.designTool as dt
import modules.utils as m
import numpy as np
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
# dt.plot3d(airplane_1)

# ------ CL - Cruise ------ #
M1 = 0.8
H1 = 10000  # m
V1 = M1 * m.get_a(H1)

MTOW1 = airplane_1['W0_guess']
rho1 = dt.atmosphere(H1)[2]
CL1_cruise = 0.95 * MTOW1 / (0.5 * rho1 * V1**2 * airplane_1['S_w'])

print(f'CL (cruise): {CL1_cruise}')

CD1_cruise, _, dragDict1 = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise, airplane_1['W0_guess'])
print(pprint.pformat(dragDict1))

m.print_drag_table(CD1_cruise, dragDict1)

## Plots
# CD x M
M1_range = np.arange(0.6, 0.9, 0.001)
m.plot_CD_x_M(M1_range, H1, CL1_cruise, airplane_1, '1')

# Drag Polar
m.drag_polar(airplane_1, CL1_cruise, '1')

## Aerodynamic Efficiency (LD)

# LD Max


# LD Cruise
# LD1_cruise = CL1_cruise/CD1_cruise


# ------------------------------------------------------------ #
# ------------------- AIRPLANE 2 ----------------------------- #
# ------------------------------------------------------------ #
print('================ AIRPLANE 2 ===========================')

airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)
# dt.plot3d(airplane_2)

# ------ CL - Cruise ------ #
M2 = 0.8
H2 = 10000  # m
MTOW2 = airplane_2['W0_guess']
V2 = M2 * m.get_a(H2)
rho2 = dt.atmosphere(H2)[2]
CL2_cruise = 0.95 * MTOW2 / (0.5 * rho2 * V2**2 * airplane_2['S_w'])

print(f'CL (cruise): {CL2_cruise}')

CD2_cruise, _, dragDict2 = dt.aerodynamics(airplane_2, M2, H2, CL2_cruise, airplane_2['W0_guess'])
print(pprint.pformat(dragDict2))

m.print_drag_table(CD2_cruise, dragDict2)

## Plots
# CD x M
M1_range = np.arange(0.6, 0.9, 0.001)
m.plot_CD_x_M(M1_range, H2, CL2_cruise, airplane_2, '2')

# Drag Polar
m.drag_polar(airplane_2, CL2_cruise, '2')

## Aerodynamic Efficiency (LD)

# LD Max


# LD Cruise
# LD2_cruise = CL2_cruise/CD2_cruise