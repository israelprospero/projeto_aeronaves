import matplotlib.pyplot as plt
import modules.designTool as dt
import numpy as np
from tabulate import tabulate

H1 = 10700
H2 = 10700

def get_a(H):
    # dt.atmosphere(H) retorna temperatura em °C, pressão, densidade, viscosidade
    T_C = dt.atmosphere(H)[0]
    T_K = T_C + 273.15   
    gamma = 1.4          
    R = 287.05           
    return np.sqrt(gamma * R * T_K)


def get_Mach_stall(airplane, W, config='takeoff', altitude=0):
    rho = dt.atmosphere(altitude)[2]       
    S = airplane['S_w']                    
    a = get_a(altitude)           

    Mach_chute = 3
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
        CD, _, _ = dt.aerodynamics(airplane, M, H, CL, Weight, highlift_config='clean') # TODO: check arguments
        LD_list.append(CL / CD)
        CL_list.append(CL)

    LD_max = max(LD_list)
    CL_LDmax = CL_list[np.argmax(LD_list)]
    CD_LDmax, _, _ = dt.aerodynamics(airplane, 0.8, 10000, CL_LDmax, airplane['W0_guess'], highlift_config='clean')
    print(f"(L/D)_max = {LD_max:.2f} at CL = {CL_LDmax:.2f}, CD = {CD_LDmax:.4f}")

def drag_polar(airplane, CL_cruise, num):
    
    labels = ['Cruise', 'Takeoff', 'Landing']
    
    Mach_stall_takeoff, _ = get_Mach_stall(airplane, airplane['W0_guess'], config='takeoff')
    Mach_stall_landing, _ = get_Mach_stall(airplane, airplane['W0_guess'], config='landing') # TODO: alterar peso para landing!!!
    configs = [
        {'M': 0.8, 'H': 10000, 'config': 'clean'},
        {'M': 1.2*Mach_stall_takeoff, 'H': 0, 'config': 'takeoff'},
        {'M': 1.3*Mach_stall_landing, 'H': 0, 'config': 'landing'}
    ]
    
    
    
    colors = ['blue', 'red', 'green']
    plt.figure()
    for label, conf, color in zip(labels, configs, colors):
        CL_list = []
        CD_list = []
        for CL in np.arange(-0.5, 3.0, 0.001):
            CD, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL, airplane['W0_guess'],
                                        highlift_config=conf['config'],
                                        n_engines_failed=1 if conf['config']=='takeoff' else 0,
                                        lg_down=1 if conf['config']=='landing' else 0)
            CL_list.append(CL)
            CD_list.append(CD)

        _, CLmax, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], 0.5, airplane['W0_guess'], highlift_config=conf['config'])
        
        mask = np.array(CL_list) <= CLmax
        plt.plot(np.array(CD_list)[mask], np.array(CL_list)[mask], label=label)
        
        if label == 'Cruise':
            CD_cruise, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL_cruise, airplane['W0_guess'])
            plt.plot(CD_cruise, CL_cruise, 'ks', label='Cruise Point A1')

        CD_clmax, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CLmax, airplane['W0_guess'], highlift_config=conf['config'])
        plt.plot(np.array(CD_list)[mask][-1], np.array(CL_list)[mask][-1], 'o', color=color)
        plt.text(np.array(CD_list)[mask][-1], np.array(CL_list)[mask][-1] + 0.05, f"CLmax {label} = {CLmax:.2f}", color=color)
            
    plt.xlabel("CD")
    plt.ylabel("CL")
    plt.title(f"Airplane {num} - Polar Drag with CLmax")
    plt.legend()
    plt.grid(True)
    plt.show() 
        
def plot_CD_x_M(M_range, H, CL, airplane, num):
    
    CD_list = []
    M_list = []
    for M in M_range:
        M_list.append(M)

        CD, _, _ = dt.aerodynamics(airplane, M, H, CL, airplane['W0_guess'])
        CD_list.append(CD)
    
    plt.plot(M_list, CD_list)
    plt.xlabel('M')
    plt.ylabel('CD')
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