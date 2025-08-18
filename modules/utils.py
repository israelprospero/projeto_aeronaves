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

def print_fuel_table(airplane, export_excel=False, filename="fuel_table.xlsx"):
    import pandas as pd

    # Dados
    total_fuel = airplane['fuel_total']
    fuel_breakdown = airplane['fuel_breakdown']
    mf_breakdown = airplane['fuel_Mf_breakdown']

    # Calcular percentuais já com trapped incluso no total
    percent_breakdown = {
        phase: f"{100 * value / total_fuel:.1f}" if isinstance(value, (int, float)) else value
        for phase, value in fuel_breakdown.items()
    }

    # Criar DataFrame formatado
    df = pd.DataFrame({
        "Mission phase": list(mf_breakdown.keys()),
        "Mf": [f"{mf:.4f}" if isinstance(mf, float) else mf for mf in mf_breakdown.values()],
        "Fuel consumed [kg]": [f"{fuel:.1f}" for fuel in fuel_breakdown.values()],
        "% of mission fuel": list(percent_breakdown.values())
    })

    # Adicionar única linha de total
    df.loc[len(df.index)] = ["TOTAL", "-", f"{total_fuel:.1f}", "100.0"]

    # Imprimir tabela formatada
    from tabulate import tabulate
    print(tabulate(df.values, headers=df.columns, tablefmt="fancy_grid"))

    # Exportar se solicitado
    if export_excel:
        df.to_excel(filename, index=False)
        print(f"\n✅ Tabela exportada para '{filename}' com sucesso.")

    return df


def plot_W0_x_ar_w(ar_w_range, airplane, num):
    
    ar_w_list = []
    W0_list = []
    Wempty_list = []
    Wfuel_list = []
    
    for k in ar_w_range:
        airplane['AR_w'] = k
        dt.geometry(airplane) # chama a função geometry para atualizar  geometria do avião com o novo ar_w antes de chamar a função 'W0'
        ar_w_list.append(k)   
        
        W0, W_empty, W_fuel, _ = dt.weight(airplane['W0_guess'], airplane['T0_guess'], airplane) #calcula o weight para cada alongamento (ar_w)
        W0_list.append(W0)
        Wempty_list.append(W_empty)
        Wfuel_list.append(W_fuel)
    
    # print(ar_w_list)
    # print(W0_list)
    # plt.figure()
    # plt.plot(ar_w_list, W0_list)
    # plt.xlabel('AR_w',fontsize=14)
    # plt.ylabel('W0',fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.title(f'Airplane {num} - W0 x AR_w')
    # plt.grid(True)
    # plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(ar_w_list, W0_list, color='navy', linewidth=2, label='W0')
    plt.plot(ar_w_list, Wempty_list, linewidth=2, label='W_empty')
    plt.plot(ar_w_list, Wfuel_list, linewidth=2, label='W_fuel')
    plt.legend()
    plt.xlabel('Wing Aspect Ratio (AR_w)', fontsize=16, fontweight='bold')
    plt.ylabel('Weights [N]', fontsize=16, fontweight='bold')
    # plt.title(f'Airplane {num} — W0 vs. AR_w', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(ar_w_list, W0_list, color='navy', linewidth=2)
    # plt.xlabel('Wing Aspect Ratio (AR_w)', fontsize=16, fontweight='bold')
    # plt.ylabel('Takeoff Weight (W0) [N]', fontsize=16, fontweight='bold')
    # # plt.title(f'Airplane {num} — W0 vs. AR_w', fontsize=18, fontweight='bold')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    # plt.tight_layout()
    # plt.show()


def plot_T0_x_Sw(airplane, Swvec):
    
    T0plot = []
    for Sw in Swvec:
        airplane['S_w'] = Sw
        dt.geometry(airplane)
        
        dt.thrust_matching(airplane['W0_guess'], airplane['T0_guess'], airplane)
        T0plot.append(airplane['T0vec'])
        
        # print(airplane['T0vec'])
        # print('\n')
        
    names = [
        "T0_takeoff",
        "T0_cruise",
        "T0_FAR25.111",
        "T0_FAR25.121a",
        "T0_FAR25.121b",
        "T0_FAR25.121c",
        "T0_FAR25.119",
        "T0_FAR25.121d"
    ]

    print(len(airplane['T0vec']))
    
    plt.figure()
    for i in range(8):
        first_terms = [lst[i] for lst in T0plot]
        plt.plot(Swvec, first_terms, label=f'{names[i]}')
        # print(first_terms)
    plt.axvline(x=airplane['deltaS_wlan'], color='r', linestyle='--', linewidth=2, label = 'landing')
    plt.legend()
    plt.xlabel('S_w (m^2)')
    plt.ylabel('T0 (N)')
    plt.show()
        