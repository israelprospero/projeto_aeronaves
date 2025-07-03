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
print('\n\n')

airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)
dt.plot3d(airplane_1)

# ------ CL - Cruise ------ #
M1 = 0.8
H1 = 10700  # m
V1 = M1 * m.get_a(H1)

MTOW1 = airplane_1['W0_guess']
rho1 = dt.atmosphere(H1)[2]
CL1_cruise = 0.95 * MTOW1 / (0.5 * rho1 * V1**2 * airplane_1['S_w'])

print('------------------------- \n')
print(f'CL (cruise): {CL1_cruise}')
print('------------------------- \n\n')

CD1_cruise, _, dragDict1 = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise, airplane_1['W0_guess'])
print(pprint.pformat(dragDict1))

m.print_drag_table(CD1_cruise, dragDict1)

# ------ CL - Landing ------ #
MLW_1 = 0.85*airplane_1['W0_guess'] # MLW - maximum landing weight - 85% of MTOW
H_landing = 0 #at sea level

M_stall_landing_1, CLmax_landing_01 = m.get_Mach_stall(airplane_1, MLW_1, 'landing', H_landing)

M_landing_1 = M_stall_landing_1*1.3
CL_landing_1 = CLmax_landing_01/(1.3**2)

CD_landing_1, CLmax_landing_1, dragDict_landing_1 = dt.aerodynamics(airplane_1, M_landing_1, H_landing, 
                                                                    CL_landing_1, MLW_1, 
                                                                    n_engines_failed=0, highlift_config='landing', 
                                                                    lg_down=1, h_ground=0, method=2,
                                                                    ind_drag_method='Nita',
                                                                    ind_drag_flap_method='Roskam')

print('------------------------- \n')
print(f'CL (landing): {CL_landing_1}')
print('------------------------- \n\n')

m.print_drag_table(CD_landing_1, dragDict_landing_1)

## Plots
# CD x M
M1_range = np.arange(0.6, 0.9, 0.001)
m.plot_CD_x_M(M1_range, H1, CL1_cruise, airplane_1, '1')

# Drag Polar
m.drag_polar(airplane_1, CL1_cruise, '1')

## Aerodynamic Efficiency (LD)

# LD Max
CL1_range = np.arange(-0.5,2.9,0.001)
m.LD_max(airplane_1, CL1_range, M1, H1,0.95 * MTOW1)
LD1_cruise = CL1_cruise/CD1_cruise
print(f'(L/D)_cruise = {LD1_cruise:.2f} at CL = {CL1_cruise:.2f}, CD = {CD1_cruise:.2f}\n\n')

## 


# ------------------------------------------------------------ #
# ------------------- AIRPLANE 2 ----------------------------- #
# ------------------------------------------------------------ #
print('================ AIRPLANE 2 ===========================')

airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)
dt.plot3d(airplane_2)

# ------ CL - Cruise ------ #
M2 = 0.8
H2 = 10700  # m
MTOW2 = airplane_2['W0_guess']
V2 = M2 * m.get_a(H2)
rho2 = dt.atmosphere(H2)[2]
CL2_cruise = 0.95 * MTOW2 / (0.5 * rho2 * V2**2 * airplane_2['S_w'])

print('------------------------- \n')
print(f'CL (cruise): {CL2_cruise}')
print('------------------------- \n\n')

CD2_cruise, _, dragDict2 = dt.aerodynamics(airplane_2, M2, H2, CL2_cruise, airplane_2['W0_guess'])
print(pprint.pformat(dragDict2))

m.print_drag_table(CD2_cruise, dragDict2)

# ------ CL - Landing ------ #
MLW_2 = 0.85*airplane_2['W0_guess'] # MLW - maximum landing weight - 85% of MTOW
H_landing = 0 #at sea level

M_stall_landing_2, CLmax_landing_02 = m.get_Mach_stall(airplane_2, MLW_2, 'landing', H_landing)

M_landing_2 = M_stall_landing_2*1.3
CL_landing_2 = CLmax_landing_02/(1.3**2)

CD_landing_2, CLmax_landing_2, dragDict_landing_2 = dt.aerodynamics(airplane_2, M_landing_2, H_landing, 
                                                                    CL_landing_2, MLW_2, 
                                                                    n_engines_failed=0, highlift_config='landing', 
                                                                    lg_down=1, h_ground=0, method=2,
                                                                    ind_drag_method='Nita',
                                                                    ind_drag_flap_method='Roskam')

print('------------------------- \n')
print(f'CL (landing): {CL_landing_2}')
print('------------------------- \n\n')

m.print_drag_table(CD_landing_2, dragDict_landing_2)

## Plots
# CD x M
M2_range = np.arange(0.6, 0.9, 0.001)
m.plot_CD_x_M(M2_range, H2, CL2_cruise, airplane_2, '2')

# Drag Polar
m.drag_polar(airplane_2, CL2_cruise, '2')

## Aerodynamic Efficiency (LD)

# LD Max
CL2_range = np.arange(-0.5,2.9,0.001)
m.LD_max(airplane_2, CL2_range, M2, H2,0.95 * MTOW2)
LD2_cruise = CL2_cruise/CD2_cruise
print(f'(L/D)_cruise = {LD2_cruise:.2f} at CL = {CL2_cruise:.2f}, CD = {CD2_cruise:.2f}')