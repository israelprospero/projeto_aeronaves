import matplotlib.pyplot as plt
import modules.designTool as dt
import numpy as np
from tabulate import tabulate

a = 331.3  # m/s^2

def CL_max(airplane):
    
    return 0.9*airplane['clmax_w']*np.cos(airplane['sweep_w'])

def drag_polar(airplane, CL_cruise, num):
    
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
        for CL in np.arange(-0.5, 3.0, 0.001):
            CD, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL, airplane['W0_guess'],
                                        highlift_config=conf['config'],
                                        n_engines_failed=1 if conf['config']=='takeoff' else 0,
                                        lg_down=1 if conf['config']=='landing' else 0)
            CL_list.append(CL)
            CD_list.append(CD)

        CLmax = CL_max(airplane)
        # _, CLmax, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], 0.5, airplane['W0_guess'], highlift_config=conf['config'])
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