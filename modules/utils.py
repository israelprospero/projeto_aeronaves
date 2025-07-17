import matplotlib.pyplot as plt
from modules import designTool as dt
import numpy as np
from tabulate import tabulate

H1 = 10700
H2 = 10700

def get_a(H):
    # dt.atmosphere(H) retorna temperatura em °C, pressão, densidade, viscosidade
    T_K = dt.atmosphere(H)[0] 
    gamma = 1.4          
    R = 287.05           
    return np.sqrt(gamma * R * T_K)


def get_Mach_stall(airplane, W, config='takeoff', altitude=0):
    rho = dt.atmosphere(altitude)[2]       
    S = airplane['S_w']                    
    a = get_a(altitude)           

    Mach_chute = 0.3
    _, CLmax, _ = dt.aerodynamics(airplane, Mach_chute, altitude, CL=1.2, W0_guess=W, highlift_config=config)
    # print(CLmax)
    
    V_stall = np.sqrt(2 * W / (rho * S * CLmax))
    Mach_stall = V_stall / a
    
    return Mach_stall, CLmax

def LD_max(airplane, CL_range, M, H, Weight):
            
    # L/D max
    CL_list = []
    LD_list = []
    for CL in CL_range:
        CD, _, _ = dt.aerodynamics(airplane, M, H, CL, Weight, highlift_config='clean') 
        LD_list.append(CL / CD)
        CL_list.append(CL)

    LD_max = max(LD_list)
    CL_LDmax = CL_list[np.argmax(LD_list)]
    CD_LDmax, _, _ = dt.aerodynamics(airplane, M, H, CL_LDmax, Weight, highlift_config='clean')
    print(f"(L/D)_max = {LD_max:.2f} at CL = {CL_LDmax:.2f}, CD = {CD_LDmax:.4f}")

def drag_polar(airplane, CL_cruise, num):
    
    labels = ['Cruise', 'Takeoff', 'Landing']
    
    Mach_stall_takeoff, _ = get_Mach_stall(airplane, airplane['W0_guess'], config='takeoff')
    Mach_stall_landing, _ = get_Mach_stall(airplane, 0.85*airplane['W0_guess'], config='landing') 
    configs = [
        {'M': 0.8, 'H': 10700, 'W': 0.95*airplane['W0_guess'], 'config': 'clean'},
        {'M': 1.2*Mach_stall_takeoff, 'H': 0, 'W': airplane['W0_guess'], 'config': 'takeoff'},
        {'M': 1.3*Mach_stall_landing, 'H': 0, 'W': 0.85*airplane['W0_guess'],'config': 'landing'}
    ]
    
    colors = ['blue', 'red', 'green']
    plt.figure()
    for label, conf, color in zip(labels, configs, colors):
        CL_list = []
        CD_list = []
        for CL in np.arange(-0.5, 3.0, 0.001):
            CD, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL, conf['W'],
                                        highlift_config=conf['config'],
                                        n_engines_failed=1 if conf['config']=='takeoff' else 0,
                                        lg_down=1 if conf['config']=='landing' else 0)
            CL_list.append(CL)
            CD_list.append(CD*1e4)

        _, CLmax, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], 0.5, conf['W'], highlift_config=conf['config'])
        
        mask = np.array(CL_list) <= CLmax
        plt.plot(np.array(CD_list)[mask], np.array(CL_list)[mask], label=label)
        
        if label == 'Cruise':
            CD_cruise, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL_cruise, conf['W'])
            CD_cruise = CD_cruise*1e4
            plt.plot(CD_cruise, CL_cruise, 'ks', label='Cruise Point A1')

        CD_clmax, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CLmax, conf['W'], highlift_config=conf['config'])
        plt.plot(np.array(CD_list)[mask][-1], np.array(CL_list)[mask][-1], 'o', color=color)
        plt.text(np.array(CD_list)[mask][-1], np.array(CL_list)[mask][-1] + 0.05, f"CLmax {label} = {CLmax:.2f}", color=color, fontsize=14)
            
    plt.xlabel("CD",fontsize=16)
    plt.ylabel("CL",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Airplane {num} - Polar Drag with CLmax")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show() 
        
def plot_CD_x_M(M_range, H, CL, airplane, num):
    
    CD_list = []
    M_list = []
    for M in M_range:
        M_list.append(M)

        CD, _, _ = dt.aerodynamics(airplane, M, H, CL, 0.95*airplane['W0_guess'])
        CD_list.append(CD*1e4)
    
    plt.figure()
    plt.plot(M_list, CD_list)
    plt.xlabel('M',fontsize=14)
    plt.ylabel('CD',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Airplane {num} - CD x M')
    plt.grid(True)
    plt.show()
    
def print_drag_table(CD, dragDict):
    
    names = list(dragDict.keys())
    values = list(dragDict.values())
    drag_list_counts = [v * 1e4 for v in values]
    perc_drag = [v / CD if name.startswith('CD') else '-' for name, v in zip(names, values)]

    table = []
    for i in range(len(names)):
        row = [names[i], values[i], drag_list_counts[i], perc_drag[i]]
        table.append(row)

    # Print table
    headers = ["Name", "Value", "Value * 10^4", "Value / CD1"]
    print(tabulate(table, headers=headers, floatfmt=".4f"))

def print_fuel_table(airplane):
    import pandas as pd
    from tabulate import tabulate

    mfs = airplane['fuel_Mf_breakdown']
    fuels = airplane['fuel_breakdown']
    percents = airplane['fuel_percent_breakdown']

    table = []
    for phase in mfs.keys():
        mf = mfs[phase]
        fuel = fuels[phase]
        percent = percents[phase]
        table.append([phase, f"{mf:.4f}", f"{fuel:.1f}", f"{percent:.1f}"])

    # Soma total
    total_fuel = airplane['fuel_total']
    total_used = airplane['fuel_total_used']

    table.append(['TOTAL (used)', '-', f"{total_used:.1f}", "100.0"])
    table.append(['TOTAL (with trapped)', '-', f"{total_fuel:.1f}", "-"])

    headers = ["Mission phase", "Mf", "Fuel consumed [kg]", "% of mission fuel"]

    # Imprimir na tela
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # Criar DataFrame
    df = pd.DataFrame(table[:-2], columns=headers)

    # Exportar para Excel
    df.to_excel("fuel_breakdown.xlsx", index=False)
    print("\nTabela exportada para 'fuel_breakdown.xlsx'.")


'''
def weight(W0_guess, T0_guess, airplane):

    # Unpacking dictionary
    W_payload = airplane['W_payload']
    W_crew = airplane['W_crew']
    range_cruise = airplane['range_cruise']

    # Set iterator
    delta = 1000

    while abs(delta) > 10:

        # We need to call fuel_weight first since it
        # calls the aerodynamics module to get Swet_f used by
        # the empty weight function
        W_fuel, Mf_cruise = fuel_weight(W0_guess, airplane, range_cruise=range_cruise, update_Mf_hist=True)

        W_empty = empty_weight(W0_guess, T0_guess, airplane)

        W0 = W_empty + W_fuel + W_payload + W_crew

        delta = W0 - W0_guess

        W0_guess = W0
        
    airplane['W0'] = W0
    airplane['W_empty'] = W_empty
    airplane['W_fuel'] = W_fuel

    # Calcular pesos brutos por fase
    phases = ['engine_start', 'taxi', 'takeoff', 'climb', 'cruise',
              'loiter', 'descent', 'altcruise', 'landing', 'trapped']
    
    Mfs = [airplane['Mf_' + p] for p in phases]
    
    W = W0/gravity
    Wfuel = W_fuel/gravity
    airplane['W_gross_total'] = W
    airplane['W_gross_fuel_total'] = Wfuel
    for phase, mf in zip(phases, Mfs):
        W_spent = W*((1 - mf))
        airplane[f'W_gross_{phase}'] = W_spent  # em kg
        W = W - W_spent

    # Calcular combustíveis consumidos
    fuel_breakdown = {}
    total_used_fuel = 0

    for phase, mf in zip(phases, Mfs):
        fuel = W0 * mf / gravity
        fuel_breakdown[phase.replace('_', ' ').title()] = fuel
        total_used_fuel += fuel

    return W0, W_empty, W_fuel, Mf_cruise
'''