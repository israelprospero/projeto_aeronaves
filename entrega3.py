import designTool as dt
import numpy as np
import pprint
import matplotlib.pyplot as plt
from tabulate import tabulate

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi
a = 331.3  # velocidade do som

# =============================================== #
# AIRPLANE 1
# =============================================== #
print('==============================')
print('AIRPLANE 1')
print('==============================')
airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)

M1 = 0.8
H1 = 10000  # m
MTOW1 = airplane_1['W0_guess']

V1 = M1 * a
rho1 = dt.atmosphere(H1)[2]
CL1_cruise_08 = 0.95 * MTOW1 / (0.5 * rho1 * V1**2 * airplane_1['S_w'])
print(f'CL (cruise): {CL1_cruise_08}')

CD1, _, dragDict1 = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise_08, airplane_1['W0_guess'])
print(pprint.pformat(dragDict1))

# Table
names1 = list(dragDict1.keys())
values1 = list(dragDict1.values())
drag_list_counts1 = [v * 1e4 for v in values1]
perc_drag1 = [v / CD1 for v in values1]

table = []
for i in range(len(names1)):
    row = [names1[i], values1[i], drag_list_counts1[i], perc_drag1[i]]
    table.append(row)

headers = ["Name", "Value", "Value * 10^4", "Value / CD1"]
print(tabulate(table, headers=headers, floatfmt=".4f"))

# CD x M
CD1_list = []
M1_list = []
for M1 in np.arange(0.6, 0.9, 0.001):
    M1_list.append(M1)
    V1 = M1 * a
    rho1 = dt.atmosphere(H1)[2]
    CL1_cruise = 0.95 * MTOW1 / (0.5 * rho1 * V1**2 * airplane_1['S_w'])
    CD1, _, _ = dt.aerodynamics(airplane_1, M1, H1, CL1_cruise, airplane_1['W0_guess'])
    CD1_list.append(CD1)

plt.plot(M1_list, CD1_list)
plt.xlabel('M')
plt.ylabel('CD')
plt.title('Airplane 1 - CD vs Mach')
plt.grid(True)
plt.show()

# -------------------------------------
# Polar Drag + CLmax
# -------------------------------------

# Cruise
CL_cruise_list = []
CD_cruise_list = []
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_cruise_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, 0.8, 10000, CL, airplane_1['W0_guess'], highlift_config='clean')
    CD_cruise_list.append(CD)

# Takeoff
CL_takeoff_list = []
CD_takeoff_list = []
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_takeoff_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, 0.3, 0, CL, airplane_1['W0_guess'], highlift_config='takeoff', n_engines_failed=1)
    CD_takeoff_list.append(CD)

# Landing
CL_app_list = []
CD_app_list = []
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_app_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_1, 0.35, 0, CL, airplane_1['W0_guess'], highlift_config='landing', lg_down=1)
    CD_app_list.append(CD)

# CLmax valores
_, CLmax_cruise, _ = dt.aerodynamics(airplane_1, 0.8, 10000, 0.5, airplane_1['W0_guess'], highlift_config='clean')
_, CLmax_takeoff, _ = dt.aerodynamics(airplane_1, 0.3, 0, 0.5, airplane_1['W0_guess'], highlift_config='takeoff')
_, CLmax_landing, _ = dt.aerodynamics(airplane_1, 0.35, 0, 0.5, airplane_1['W0_guess'], highlift_config='landing')

print(f"\nCLmax (Airplane 1):")
print(f"Cruise  : {CLmax_cruise:.3f}")
print(f"Takeoff : {CLmax_takeoff:.3f}")
print(f"Landing : {CLmax_landing:.3f}")

# 
cruise_mask = np.array(CL_cruise_list) <= CLmax_cruise
takeoff_mask = np.array(CL_takeoff_list) <= CLmax_takeoff
landing_mask = np.array(CL_app_list) <= CLmax_landing

plt.figure()

plt.plot(np.array(CD_cruise_list)[cruise_mask], np.array(CL_cruise_list)[cruise_mask], label='Cruise')
plt.plot(np.array(CD_takeoff_list)[takeoff_mask], np.array(CL_takeoff_list)[takeoff_mask], label='Takeoff')
plt.plot(np.array(CD_app_list)[landing_mask], np.array(CL_app_list)[landing_mask], label='Landing')


plt.plot(np.array(CD_cruise_list)[cruise_mask][-1], np.array(CL_cruise_list)[cruise_mask][-1], 'bo')
plt.text(np.array(CD_cruise_list)[cruise_mask][-1], np.array(CL_cruise_list)[cruise_mask][-1] + 0.05, f'CLmax Cruise = {CLmax_cruise:.2f}', color='blue')
plt.plot(np.array(CD_takeoff_list)[takeoff_mask][-1], np.array(CL_takeoff_list)[takeoff_mask][-1], 'ro')
plt.text(np.array(CD_takeoff_list)[takeoff_mask][-1], np.array(CL_takeoff_list)[takeoff_mask][-1] + 0.05, f'CLmax Takeoff = {CLmax_takeoff:.2f}', color='red')
plt.plot(np.array(CD_app_list)[landing_mask][-1], np.array(CL_app_list)[landing_mask][-1], 'go')
plt.text(np.array(CD_app_list)[landing_mask][-1], np.array(CL_app_list)[landing_mask][-1] + 0.05, f'CLmax Landing = {CLmax_landing:.2f}', color='green')

plt.xlabel('CD')
plt.ylabel('CL')
plt.title('Airplane 1 - Polar Drag with CLmax')
plt.grid(True)
plt.legend()
plt.show()

# Velocidades de estol
S1 = airplane_1['S_w']
rho_cruise = dt.atmosphere(10000)[2]
rho_takeoff = dt.atmosphere(0)[2]
rho_landing = dt.atmosphere(0)[2]

V_stall_cruise = np.sqrt(2 * MTOW1 / (rho_cruise * S1 * CLmax_cruise))
V_stall_takeoff = np.sqrt(2 * MTOW1 / (rho_takeoff * S1 * CLmax_takeoff))
V_stall_landing = np.sqrt(2 * MTOW1 / (rho_landing * S1 * CLmax_landing))

print(f"\nVelocidades de Estol (Airplane 1):")
print(f"Cruise  : {V_stall_cruise:.2f} m/s")
print(f"Takeoff : {V_stall_takeoff:.2f} m/s")
print(f"Landing : {V_stall_landing:.2f} m/s")

# =============================================== #
# AIRPLANE 2 (com mesmo processo)
# =============================================== #
print('\n==============================')
print('AIRPLANE 2')
print('==============================')
airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)

MTOW2 = airplane_2['W0_guess']

# Cruise
CL_cruise_list = []
CD_cruise_list = []
L_D_list = list()
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_cruise_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, 0.8, 10000, CL, airplane_2['W0_guess'], highlift_config='clean')
    CD_cruise_list.append(CD)
    L_D_list.append(CL/CD)

# Takeoff
CL_takeoff_list = []
CD_takeoff_list = []
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_takeoff_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, 0.3, 0, CL, airplane_2['W0_guess'], highlift_config='takeoff', n_engines_failed=1)
    CD_takeoff_list.append(CD)

# Landing
CL_app_list = []
CD_app_list = []
for CL in np.arange(-0.5, 3.0, 0.001):
    CL_app_list.append(CL)
    CD, _, _ = dt.aerodynamics(airplane_2, 0.35, 0, CL, airplane_2['W0_guess'], highlift_config='landing', lg_down=1)
    CD_app_list.append(CD)

# CLmax valores
_, CLmax_cruise2, _ = dt.aerodynamics(airplane_2, 0.8, 10000, 0.5, airplane_2['W0_guess'], highlift_config='clean')
_, CLmax_takeoff2, _ = dt.aerodynamics(airplane_2, 0.3, 0, 0.5, airplane_2['W0_guess'], highlift_config='takeoff')
_, CLmax_landing2, _ = dt.aerodynamics(airplane_2, 0.35, 0, 0.5, airplane_2['W0_guess'], highlift_config='landing')

print("\nCLmax (Airplane 2):")
print(f"Cruise  : {CLmax_cruise2:.3f}")
print(f"Takeoff : {CLmax_takeoff2:.3f}")
print(f"Landing : {CLmax_landing2:.3f}")

cruise_mask = np.array(CL_cruise_list) <= CLmax_cruise2
takeoff_mask = np.array(CL_takeoff_list) <= CLmax_takeoff2
landing_mask = np.array(CL_app_list) <= CLmax_landing2

plt.figure()

plt.plot(np.array(CD_cruise_list)[cruise_mask], np.array(CL_cruise_list)[cruise_mask], label='Cruise')
plt.plot(np.array(CD_takeoff_list)[takeoff_mask], np.array(CL_takeoff_list)[takeoff_mask], label='Takeoff')
plt.plot(np.array(CD_app_list)[landing_mask], np.array(CL_app_list)[landing_mask], label='Landing')

plt.plot(np.array(CD_cruise_list)[cruise_mask][-1], np.array(CL_cruise_list)[cruise_mask][-1], 'bo')
plt.text(np.array(CD_cruise_list)[cruise_mask][-1], np.array(CL_cruise_list)[cruise_mask][-1] + 0.05, f'CLmax Cruise = {CLmax_cruise2:.2f}', color='blue')
plt.plot(np.array(CD_takeoff_list)[takeoff_mask][-1], np.array(CL_takeoff_list)[takeoff_mask][-1], 'ro')
plt.text(np.array(CD_takeoff_list)[takeoff_mask][-1], np.array(CL_takeoff_list)[takeoff_mask][-1] + 0.05, f'CLmax Takeoff = {CLmax_takeoff2:.2f}', color='red')
plt.plot(np.array(CD_app_list)[landing_mask][-1], np.array(CL_app_list)[landing_mask][-1], 'go')
plt.text(np.array(CD_app_list)[landing_mask][-1], np.array(CL_app_list)[landing_mask][-1] + 0.05, f'CLmax Landing = {CLmax_landing2:.2f}', color='green')

plt.xlabel('CD')
plt.ylabel('CL')
plt.title('Airplane 2 - Polar Drag with CLmax')
plt.grid(True)
plt.legend()
plt.show()

S2 = airplane_2['S_w']
rho_cruise2 = dt.atmosphere(10000)[2]
rho_takeoff2 = dt.atmosphere(0)[2]
rho_landing2 = dt.atmosphere(0)[2]

V_stall_cruise2 = np.sqrt(2 * MTOW2 / (rho_cruise2 * S2 * CLmax_cruise2))
V_stall_takeoff2 = np.sqrt(2 * MTOW2 / (rho_takeoff2 * S2 * CLmax_takeoff2))
V_stall_landing2 = np.sqrt(2 * MTOW2 / (rho_landing2 * S2 * CLmax_landing2))

print(f"\nVelocidades de Estol (Airplane 2):")
print(f"Cruise  : {V_stall_cruise2:.2f} m/s")
print(f"Takeoff : {V_stall_takeoff2:.2f} m/s")
print(f"Landing : {V_stall_landing2:.2f} m/s")

# === Eficiência aerodinâmica máxima em cruzeiro (Airplane 1) ===
M = 0.8
H = 10000  # m

CL_list = []
LD_list = []

for CL in np.arange(0.1, 2.5, 0.001):  # evita valores muito baixos de CL
    CD, _, _ = dt.aerodynamics(airplane_1, M, H, CL, airplane_1['W0_guess'], highlift_config='clean')
    LD = CL / CD
    CL_list.append(CL)
    LD_list.append(LD)

LD_max = max(LD_list)
CL_LDmax = CL_list[np.argmax(LD_list)]
CD_LDmax, _, _ = dt.aerodynamics(airplane_1, M, H, CL_LDmax, airplane_1['W0_guess'], highlift_config='clean')

print(f"\nEficiência aerodinâmica máxima (Airplane 1 - Cruzeiro):")
print(f"(L/D)_max = {LD_max:.2f} para CL = {CL_LDmax:.3f} e CD = {CD_LDmax:.4f}")

      
# === Eficiência aerodinâmica máxima em cruzeiro (Airplane 2) ===
M = 0.8
H = 10000  # m

CL_list_2 = []
LD_list_2 = []

for CL in np.arange(0.1, 2.5, 0.001):  # evita CL muito baixos
    CD, _, _ = dt.aerodynamics(airplane_2, M, H, CL, airplane_2['W0_guess'], highlift_config='clean')
    LD = CL / CD
    CL_list_2.append(CL)
    LD_list_2.append(LD)

LD_max_2 = max(LD_list_2)
CL_LDmax_2 = CL_list_2[np.argmax(LD_list_2)]
CD_LDmax_2, _, _ = dt.aerodynamics(airplane_2, M, H, CL_LDmax_2, airplane_2['W0_guess'], highlift_config='clean')

print(f"\nEficiência aerodinâmica máxima (Airplane 2 - Cruzeiro):")
print(f"(L/D)_max = {LD_max_2:.2f} para CL = {CL_LDmax_2:.3f} e CD = {CD_LDmax_2:.4f}")

