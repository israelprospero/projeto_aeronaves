import designTool as dt
import numpy as np
import pprint
import matplotlib.pyplot as plt
from tabulate import tabulate

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi
a = 331.3 # velocidade do som

# =============================================== #
# AIRPLANE 1
# =============================================== #
print('==============================')
print('AIRPLANE 1')
print('==============================')
airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)

M1 = 0.8
H1 = 10000 # m
MTOW1 = airplane_1['W0_guess']

V1 = M1*a
rho1 = dt.atmosphere(H1)[2]
CL1_cruise_08 = 0.95*MTOW1/(0.5*rho1*V1**2 * airplane_1['S_w'])
print(f'CL (cruise): {CL1_cruise_08}')

CD1, _, dragDict1 = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise_08, airplane_1['W0_guess'])
print(pprint.pformat(dragDict1))

## Table
# Lists
names1 = list(dragDict1.keys())
values1 = list(dragDict1.values())
drag_list_counts1 = [v * 1e4 for v in values1]
perc_drag1 = [v / CD1 for v in values1]

table = []
for i in range(len(names1)):
    row = [names1[i], values1[i], drag_list_counts1[i], perc_drag1[i]]
    table.append(row)

# Print table
headers = ["Name", "Value", "Value * 10^4", "Value / CD1"]
print(tabulate(table, headers=headers, floatfmt=".4f"))

# CD x M
CD1_list = []
M1_list = []
for M1 in np.arange(0.6, 0.9, 0.001):
    M1_list.append(M1)
    V1 = M1*a
    rho1 = dt.atmosphere(H1)[2]
    CL1_cruise = 0.95*MTOW1/(0.5*rho1*V1**2 * airplane_1['S_w'])

    CD1, _, dragDict1 = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise, airplane_1['W0_guess'])
    CD1_list.append(CD1)
    
plt.plot(M1_list, CD1_list)
plt.xlabel('M')
plt.ylabel('CD')
plt.title('Airplane 1')
plt.grid(True)
plt.show()

# -------------------------------------
# Polar Drag - Use CL1_cruise_08
# -------------------------------------

# Cruise
M = 0.8
H = 10000 # m
lg_down = 0

CL_cruise_list = []
CD_cruise_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_cruise_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, M, H, CL, airplane_1['W0_guess'])
    CD_cruise_list.append(CD)
    
# Takeoff Climb
# How to find the stall speed?
M = 0.3
H = 0
lg_down = 0

CL_takeoff_list = []
CD_takeoff_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_takeoff_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, M, H, CL, airplane_1['W0_guess'], n_engines_failed=1, highlift_config='takeoff')
    CD_takeoff_list.append(CD)

# Approach
M = 0.35
H = 0

CL_app_list = []
CD_app_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_app_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, M, H, CL, airplane_1['W0_guess'], n_engines_failed=0, highlift_config='landing', lg_down=1)
    CD_app_list.append(CD)

CD1_cruise_08, _, _ = dt.aerodynamics(airplane_1, M, H, CL1_cruise_08, airplane_1['W0_guess'])
plt.plot(CD_cruise_list, CL_cruise_list, label='Cruise')
plt.plot(CD1_cruise_08, CL1_cruise_08, 'ro')
plt.plot(CD_takeoff_list, CL_takeoff_list, label='Takeoff')
plt.plot(CD_app_list, CL_app_list, label='Approach')
plt.xlabel('CD')
plt.ylabel('CL')
plt.title('Airplane 1')
plt.grid(True)
plt.legend()
plt.show()

# =============================================== #
# AIRPLANE 2
# =============================================== #
print('==============================')
print('AIRPLANE 2')
print('==============================')
airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)

M2 = 0.8
H2 = 10000 # m
MTOW2 = airplane_2['W0_guess']
a = 331.3 # velocidade do som

V2 = M2*a
rho2 = dt.atmosphere(H2)[2]
CL2_cruise_08 = 0.95*MTOW2/(0.5*rho2*V2**2 * airplane_2['S_w'])
print(f'CL (cruise): {CL2_cruise_08}')

CD2, _, dragDict2 = dt.aerodynamics(airplane_2, M2, H2, CL2_cruise_08, airplane_2['W0_guess'])

print(pprint.pformat(dragDict2))

## Table
# Lists
names2 = list(dragDict2.keys())
values2 = list(dragDict2.values())
drag_list_counts2 = [v * 1e4 for v in values2]
perc_drag2 = [v / CD2 for v in values2]

table = []
for i in range(len(names2)):
    row = [names2[i], values2[i], drag_list_counts2[i], perc_drag2[i]]
    table.append(row)

# Print table
headers = ["Name", "Value", "Value * 10^4", "Value / CD2"]
print(tabulate(table, headers=headers, floatfmt=".4f"))

# CD x M
CD2_list = []
M2_list = []
for M2 in np.arange(0.6, 0.9, 0.001):
    M2_list.append(M2)
    V2 = M2*a
    rho2 = dt.atmosphere(H2)[2]
    CL2_cruise = 0.95*MTOW2/(0.5*rho2*V2**2 * airplane_2['S_w'])

    CD2, _, dragDict2 = dt.aerodynamics(airplane_2, M2, H2, CL2_cruise, airplane_2['W0_guess'])
    CD2_list.append(CD2)
    
plt.plot(M2_list, CD2_list)
plt.xlabel('M')
plt.ylabel('CD')
plt.title('Airplane 2')
plt.grid(True)
plt.show()

# -------------------------------------
# Polar Drag - Use CL2_cruise_08
# -------------------------------------

# Cruise
M = 0.8
H = 10000 # m
lg_down = 0

CL_cruise_list = []
CD_cruise_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_cruise_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, M, H, CL, airplane_2['W0_guess'])
    CD_cruise_list.append(CD)
    
# Takeoff Climb
# How to find the stall speed?
M = 0.3
H = 0
lg_down = 0

CL_takeoff_list = []
CD_takeoff_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_takeoff_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, M, H, CL, airplane_2['W0_guess'], n_engines_failed=1, highlift_config='takeoff')
    CD_takeoff_list.append(CD)

# Approach
M = 0.35
H = 0

CL_app_list = []
CD_app_list = []
for CL in np.arange(-0.5, 1.5, 0.001):
    CL_app_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, M, H, CL, airplane_2['W0_guess'], n_engines_failed=0, highlift_config='landing', lg_down=1)
    CD_app_list.append(CD)

CD2_cruise_08, _, _ = dt.aerodynamics(airplane_2, M, H, CL1_cruise_08, airplane_2['W0_guess'])
plt.plot(CD_cruise_list, CL_cruise_list, label='Cruise')
plt.plot(CD2_cruise_08, CL2_cruise_08, 'ro')
plt.plot(CD_takeoff_list, CL_takeoff_list, label='Takeoff')
plt.plot(CD_app_list, CL_app_list, label='Approach')
plt.xlabel('CD')
plt.ylabel('CL')
plt.title('Airplane 2')
plt.grid(True)
plt.legend()
plt.show()