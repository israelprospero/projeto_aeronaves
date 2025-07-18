import modules.designTool as dt
import modules.utils as m
import numpy as np
import pprint
import matplotlib.pyplot as plt
from tabulate import tabulate

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi    

airplane = dt. standard_airplane ('fokker100')
dt.geometry(airplane)

# dt.plot3d(airplane)

W0_guess = 40000*dt.gravity
T0_guess = 0.3*W0_guess
W0, W_empty, W_fuel, _, Mf_list = dt.weight(W0_guess,T0_guess, airplane)

print('Weights in kgf:')
print('W0: %d '%(W0/dt.gravity))
print('W_empty : %d '%(W_empty/dt.gravity))
print('W_fuel : %d '%(W_fuel /dt.gravity))
print('W_payload : %d '%(airplane['W_payload']/dt.gravity))
print('W_crew : %d '%(airplane['W_crew']/dt.gravity))

Weight_list = [W_fuel/dt.gravity, airplane['W_payload']/dt.gravity, airplane['W_crew']/dt.gravity]

print('breakdown :')
for key in ['W_w','W_h','W_v','W_f','W_nlg','W_mlg','W_eng','W_allelse']:
    print(key +': %d '%(airplane[key]/dt.gravity))
    
    Weight_list.append(airplane[key])

Key_list = ['Combustível', 'Payload', 'Tripulação', 'Wing', 'HT', 'VT', 'Fuselagem', 'NLG', 'MLG', 'Mtores', 'allelse']

plt.figure(figsize=(8, 8))
plt.pie(Weight_list, labels=Key_list, autopct='%1.1f%%', startangle=140)
plt.title('Distribuição de Pesos da Aeronave (kgf)')
plt.axis('equal')
plt.tight_layout()
plt.show()

## Table
flight_phases = ["Engine start and warm-up", "Taxi", "Takeoff", "Climb", "Cruise", "Loiter", "Descent", "Alternate cruise", "Landing", "Trapped", "Total"]

Fuel_consumed_list = []
Perc_list = []
Wf = W0
for k in range(len(flight_phases)-2):
    Wf_next = Mf_list[k]*Wf
    fuel_consumed = Wf - Wf_next
    perc = fuel_consumed/W_fuel
    
    Fuel_consumed_list.append(fuel_consumed/dt.gravity)
    Perc_list.append(perc*100)
    
    Wf = Wf_next
    # print(k)
    
trapped_fuel = 0.06*W_fuel
Fuel_consumed_list.append(trapped_fuel/dt.gravity)
Mf_list.append('-')
Perc_list.append(0.06)

total_fuel_consumption = sum(Fuel_consumed_list)
Mf_list.append('-')
Fuel_consumed_list.append(total_fuel_consumption)
Perc_list.append(total_fuel_consumption/W0)
    
rows = list(zip(flight_phases, Mf_list, Fuel_consumed_list, Perc_list))
print(tabulate(rows, headers=['Mission Phase', 'Mf', 'Fuel consumed', 'perc. of mission fuel'], tablefmt="grid"))

# Plot W0 x AR_w
AR_w_range = np.arange(6, 14, 0.1)
W0_list = []
for AR in AR_w_range:
    airplane['AR_w'] = AR
    dt.geometry(airplane)
    
    W0_guess = 40000*dt.gravity
    T0_guess = 0.3*W0_guess
    W0, _, _, _, _ = dt.weight(W0_guess,T0_guess, airplane)
    
    W0_list.append(W0)

plt.figure()
plt.plot(AR_w_range, W0_list)
plt.xlabel('AR',fontsize=14)
plt.ylabel('MTOW',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f'Airplane 1')
plt.grid(True)
plt.show()


