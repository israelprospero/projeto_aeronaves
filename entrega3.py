
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

# Table individual (Airplane 1)
names1 = list(dragDict1.keys())
values1 = list(dragDict1.values())
drag_list_counts1 = [v * 1e4 for v in values1]
perc_drag1 = [v / CD1 if CD1 != 0 else 0 for v in values1]

table1 = []
for i in range(len(names1)):
    row = [names1[i], values1[i], drag_list_counts1[i], perc_drag1[i]]
    table1.append(row)

headers = ["Name", "Value", "Value * 10^4", "Value / CD1"]
print(tabulate(table1, headers=headers, floatfmt=".4f"))

# =============================================== #
# AIRPLANE 2
# =============================================== #
print('==============================')
print('AIRPLANE 2')
print('==============================')
airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)

M2 = 0.8
H2 = 10000  # m
MTOW2 = airplane_2['W0_guess']

V2 = M2 * a
rho2 = dt.atmosphere(H2)[2]
CL2_cruise_08 = 0.95 * MTOW2 / (0.5 * rho2 * V2**2 * airplane_2['S_w'])
print(f'CL (cruise): {CL2_cruise_08}')

CD2, _, dragDict2 = dt.aerodynamics(airplane_2, M2, H2, CL2_cruise_08, airplane_2['W0_guess'])
print(pprint.pformat(dragDict2))

# Table individual (Airplane 2)
names2 = list(dragDict2.keys())
values2 = list(dragDict2.values())
drag_list_counts2 = [v * 1e4 for v in values2]
perc_drag2 = [v / CD2 if CD2 != 0 else 0 for v in values2]

table2 = []
for i in range(len(names2)):
    row = [names2[i], values2[i], drag_list_counts2[i], perc_drag2[i]]
    table2.append(row)

headers = ["Name", "Value", "Value * 10^4", "Value / CD2"]
print(tabulate(table2, headers=headers, floatfmt=".4f"))

# =============================================== #
# TABELA COMPARATIVA LADO A LADO
# =============================================== #
print("\n==============================")
print("TABELA COMPARATIVA - AIRPLANE 1 x 2")
print("==============================")

# Unir nomes únicos das duas listas
all_keys = sorted(set(names1 + names2))

# Construir linhas lado a lado
comparative_table = []
for key in all_keys:
    v1 = dragDict1.get(key, 0.0)
    v2 = dragDict2.get(key, 0.0)
    row = [
        key,
        v1, v2,
        v1 * 1e4, v2 * 1e4,
        v1 / CD1 if CD1 != 0 else 0,
        v2 / CD2 if CD2 != 0 else 0
    ]
    comparative_table.append(row)

headers_comp = [
    "Name", 
    "Value 1", "Value 2",
    "1 * 1e4", "2 * 1e4", 
    "1 / CD1", "2 / CD2"
]

print(tabulate(comparative_table, headers=headers_comp, floatfmt=".4f"))

# =============================================== #
# CLmax e POLAR até CLmax com anotação (Airplane 1 e 2)
# =============================================== #
for i, airplane in enumerate([airplane_1, airplane_2], start=1):
    print(f"\n==== CLmax e Polar - Airplane {i} ====")
    labels = ['Cruise', 'Takeoff', 'Landing']
    configs = [
        {'M': 0.8, 'H': 10000, 'config': 'clean'},
        {'M': 0.3, 'H': 0, 'config': 'takeoff'},
        {'M': 0.35, 'H': 0, 'config': 'landing'}
    ]
    colors = ['blue', 'red', 'green']
    plt.figure()
    for label, conf, color in zip(labels, configs, colors):
        CL_list = []
        CD_list = []
        for CL in np.arange(0.1, 3.0, 0.001):
            CD, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL, airplane['W0_guess'],
                                        highlift_config=conf['config'],
                                        n_engines_failed=1 if conf['config']=='takeoff' else 0,
                                        lg_down=1 if conf['config']=='landing' else 0)
            CL_list.append(CL)
            CD_list.append(CD)

        _, CLmax, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], 0.5, airplane['W0_guess'], highlift_config=conf['config'])
        mask = np.array(CL_list) <= CLmax
        plt.plot(np.array(CD_list)[mask], np.array(CL_list)[mask], label=label)

        CD_clmax, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CLmax, airplane['W0_guess'], highlift_config=conf['config'])
        plt.plot(CD_clmax, CLmax, 'o', color=color)
        plt.text(CD_clmax, CLmax + 0.05, f"CLmax {label} = {CLmax:.2f}", color=color)

    plt.xlabel("CD")
    plt.ylabel("CL")
    plt.title(f"Airplane {i} - Polar Drag with CLmax")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================== #
# Velocidade de Estol e Eficiência Aerodinâmica Máxima
# =============================================== #
for i, (airplane, CLmax_vals, MTOW) in enumerate([
    (airplane_1, ['clean', 'takeoff', 'landing'], MTOW1),
    (airplane_2, ['clean', 'takeoff', 'landing'], MTOW2)
], start=1):
    print(f"\n==== Velocidade de Estol e (L/D)max - Airplane {i} ====")
    S = airplane['S_w']
    for config, label, H in zip(CLmax_vals, ['Cruise', 'Takeoff', 'Landing'], [10000, 0, 0]):
        _, CLmax, _ = dt.aerodynamics(airplane, 0.5, H, 0.5, airplane['W0_guess'], highlift_config=config)
        rho = dt.atmosphere(H)[2]
        V_stall = np.sqrt(2 * MTOW / (rho * S * CLmax))
        print(f"{label} - CLmax: {CLmax:.2f}, V_stall: {V_stall:.2f} m/s")

    # L/D max
    CL_list = []
    LD_list = []
    for CL in np.arange(0.1, 2.5, 0.001):
        CD, _, _ = dt.aerodynamics(airplane, 0.8, 10000, CL, airplane['W0_guess'], highlift_config='clean')
        LD_list.append(CL / CD)
        CL_list.append(CL)

    LD_max = max(LD_list)
    CL_LDmax = CL_list[np.argmax(LD_list)]
    CD_LDmax, _, _ = dt.aerodynamics(airplane, 0.8, 10000, CL_LDmax, airplane['W0_guess'], highlift_config='clean')
    print(f"(L/D)_max = {LD_max:.2f} at CL = {CL_LDmax:.2f}, CD = {CD_LDmax:.4f}")
