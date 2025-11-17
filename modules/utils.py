import matplotlib.pyplot as plt
from modules import designTool as dt
import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import copy

# --- Funções H1, H2, get_a, get_Mach_stall, LD_max ---
H1 = 10700
H2 = 10700

def get_a(H):
    # dt.atmosphere(H) retorna temperatura em K, pressão, densidade, viscosidade
    T_K = dt.atmosphere(H)[0] 
    gamma = 1.4          
    R = 287.05           
    return np.sqrt(gamma * R * T_K)


def get_Mach_stall(airplane, W, config='takeoff', altitude=0):
    rho = dt.atmosphere(altitude)[2]       
    S = airplane['S_w']                    
    a = get_a(altitude)           

    Mach_chute = 0.3
    _, CLmax, _ = dt.aerodynamics(airplane, Mach_chute, altitude, CL=1.2, highlift_config=config)
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
    #print(f"(L/D)_max = {LD_max:.2f} at CL = {CL_LDmax:.2f}, CD = {CD_LDmax:.4f}")


# ####################################################################
# ###               FUNÇÃO drag_polar                  ###
# ####################################################################

def drag_polar(airplane, CL_cruise, num, save_path=None, airplane_comp=None, label_comp="Comparação"):
    """
    Plota as polares de arrasto. Opcionalmente compara com uma segunda aeronave.
    
    Args:
        airplane (dict): Dicionário da aeronave principal.
        CL_cruise (float): CL de cruzeiro da aeronave principal.
        num (int): Número/Identificador da aeronave principal.
        save_path (str, optional): Caminho para salvar o PDF.
        airplane_comp (dict, optional): Dicionário da aeronave de comparação.
        label_comp (str): Nome para aparecer na legenda da aeronave de comparação.
    """
    import matplotlib.ticker as ticker

    colors = {'clean': '#0033A0', 'takeoff': '#D40000', 'landing': '#007A33'} # Azul, Vermelho, Verde
    labels_fase = {'clean': 'Cruzeiro', 'takeoff': 'Decolagem', 'landing': 'Pouso'}

    planes_to_plot = [(airplane, f"Avião {num}", '-', 1.0)]
    
    if airplane_comp is not None:
        planes_to_plot.append((airplane_comp, label_comp, '--', 0.7))

    plt.figure(figsize=(11, 8))
    ax = plt.gca()

    for current_plane, plane_label, ls, alpha_val in planes_to_plot:
        
        Mach_stall_to, _ = get_Mach_stall(current_plane, current_plane['W0'], config='takeoff')
        Mach_stall_ld, _ = get_Mach_stall(current_plane, 0.85*current_plane['W0'], config='landing')
        
        configs = [
            {'M': current_plane['Mach_cruise'], 'H': current_plane['altitude_cruise'], 'config': 'clean'},
            {'M': 1.2*Mach_stall_to, 'H': 0, 'config': 'takeoff'},
            {'M': 1.3*Mach_stall_ld, 'H': 0, 'config': 'landing'}
        ]

        for conf in configs:
            cfg_name = conf['config']
            color = colors[cfg_name]

            n_engines_failed = 1 if cfg_name == 'takeoff' else 0
            lg_down = 1 if cfg_name == 'landing' else 0

            CL_list = []
            CD_list = []
            
            for CL in np.arange(-0.5, 3.2, 0.02):
                CD, _, _ = dt.aerodynamics(current_plane, conf['M'], conf['H'], CL,
                                           highlift_config=cfg_name,
                                           n_engines_failed=n_engines_failed,
                                           lg_down=lg_down)
                CL_list.append(CL)
                CD_list.append(CD)
            
            _, CLmax, _ = dt.aerodynamics(current_plane, conf['M'], conf['H'], 0.5, highlift_config=cfg_name)
            mask = np.array(CL_list) <= CLmax
            
            final_label = f"{plane_label} - {labels_fase[cfg_name]}"
            
            plt.plot(np.array(CD_list)[mask], np.array(CL_list)[mask], 
                     color=color, linestyle=ls, linewidth=2, alpha=alpha_val, label=final_label)

            CD_clmax, _, _ = dt.aerodynamics(current_plane, conf['M'], conf['H'], CLmax, 
                                             highlift_config=cfg_name,
                                             n_engines_failed=n_engines_failed,
                                             lg_down=lg_down)
            
            plt.plot(CD_clmax, CLmax, 'o', color=color, markersize=6, alpha=alpha_val)
            
            if ls == '-': 
                plt.text(CD_clmax + 0.005, CLmax, f"{CLmax:.2f}", 
                         color=color, fontsize=11, fontweight='bold', va='center')

            if cfg_name == 'clean':
                if ls == '-':
                    target_CL = CL_cruise
                    marker_style = 's'
                    marker_size = 9
                else:
                    rho = dt.atmosphere(current_plane['altitude_cruise'])[2]
                    v_cr = current_plane['Mach_cruise'] * get_a(current_plane['altitude_cruise'])
                    w_cr = 0.95 * current_plane['W0']
                    target_CL = (2*w_cr)/(rho * current_plane['S_w'] * v_cr**2)
                    marker_style = 'D'
                    marker_size = 7

                CD_cr_val, _, _ = dt.aerodynamics(current_plane, conf['M'], conf['H'], target_CL, highlift_config='clean')
                plt.plot(CD_cr_val, target_CL, marker_style, color=color, markersize=marker_size, alpha=alpha_val)
                
                if ls == '-': 
                     plt.text(CD_cr_val + 0.005, target_CL, f"Cruise (CL={target_CL:.2f})", 
                         color=color, fontsize=11, va='center')

    ax.minorticks_on()
    
    plt.grid(which='major', linestyle='-', linewidth='0.8', color='gray', alpha=0.6)
    
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)

    plt.xlabel("Coeficiente de Arrasto ($C_D$)", fontsize=14, fontweight='bold')
    plt.ylabel("Coeficiente de Sustentação ($C_L$)", fontsize=14, fontweight='bold')
    #plt.title(f"Comparativo de Polares de Arrasto", fontsize=16, fontweight='bold')
    
    plt.ylim(bottom=-0.5)
    plt.xlim(left=0)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9, shadow=True)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"--- Gráfico comparativo salvo em: {save_path} ---")
        plt.show()
    else:
        plt.show()
        

def plot_CD_x_M(M_range, H, CL, airplane, W, num):
    
    CD_list = []
    M_list = []
    for M in M_range:
        M_list.append(M)
        
        rho = dt.atmosphere(H)[2] 
        V = M * get_a(H)
        CL = W / (0.5 * rho * V**2 * airplane['S_w'])
        CD, _, _ = dt.aerodynamics(airplane, M, H, CL)
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
    

def plot_T0_x_Sw(airplane, Swvec, W0_guess, T0_guess, op_point=None, savepath=None):
    """
    Plota as curvas de empuxo requerido vs área alar para diferentes condições.
    """
    
    airplane_copy = copy.deepcopy(airplane)

    T0plot = []
    deltaS_landing_last = None

    for Sw in Swvec:
        airplane_copy['S_w'] = float(Sw)
        dt.geometry(airplane_copy)

        dt.thrust_matching(W0_guess, T0_guess, airplane_copy)
        T0plot.append(airplane_copy['T0vec'])
        deltaS_landing_last = airplane_copy.get('deltaS_wlan', None)

    names = [
        "T0_takeoff", "T0_cruise", "T0_FAR25.111", "T0_FAR25.121a",
        "T0_FAR25.121b", "T0_FAR25.121c", "T0_FAR25.119", "T0_FAR25.121d"
    ]

    plt.rcParams.update({
        "font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5.2))
    T0plot = np.array(T0plot)

    for i, name in enumerate(names):
        ax.plot(Swvec, T0plot[:, i], linewidth=2, label=name)

    if deltaS_landing_last is not None:
        ax.axvline(x=deltaS_landing_last, color="tab:red", linestyle="--",
                   linewidth=1.8, label="Landing")

    ax.set_xlabel(r"$S_w$  [m$^2$]",fontsize=16)
    ax.set_ylabel(r"$T_0$  [N]",fontsize=16)
    ax.legend(ncol=2, frameon=False, fontsize=14, loc="best")

    if op_point is not None:
        S_op, T_op = op_point
        ax.scatter([S_op], [T_op], s=120, marker="o", facecolor="white",
                   edgecolor="black", linewidths=1.8, zorder=5)
        ax.scatter([S_op], [T_op], s=28, marker="o", color="black", zorder=6)
        ax.axhline(y=T_op, linestyle=":", linewidth=1.5, color="gray")
        ax.axvline(x=S_op, linestyle=":", linewidth=1.5, color="gray")
        ax.annotate(f"({S_op:.0f} m², {T_op:,.0f} N)",
                    xy=(S_op, T_op), xytext=(12, 14), textcoords="offset points",
                    fontsize=11,
                    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.7))

    fig.tight_layout()
    if savepath:
        # Salva em PDF se savepath for fornecido
        if savepath.endswith('.pdf'):
            fig.savefig(savepath, dpi=300, bbox_inches="tight", format='pdf')
            print(f"--- Gráfico T0xSw salvo em: {savepath} ---")
        else:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
            print(f"--- Gráfico T0xSw salvo em: {savepath} ---")
        plt.close(fig)
    else:
        plt.show()

# --- Funções plot_W0_x_Sw e plot_W0_x_sweep (sem mudanças) ---

def plot_W0_x_Sw(airplane, Swvec, sweep_wing_v, flap_type_v, W0_guess, T0_guess):
    # Lista para armazenar resultados
    results = []

    sweep_deg_v = sweep_wing_v * 180/np.pi

    figs = []

    # Loop sobre cada config de flap
    for flap_type in flap_type_v:
        airplane['flap_type'] = flap_type
        dt.geometry(airplane)

        # Loop para coletar os resultados
        for sweep_deg, sweep_wing in zip(sweep_deg_v, sweep_wing_v):
            W0plot = []  # reinicia para cada curva
            airplane['sweep_w'] = sweep_wing
            dt.geometry(airplane)

            # Loop sobre cada área de asa
            for Sw in Swvec:
                airplane['S_w'] = Sw
                dt.geometry(airplane)

                dt.thrust_matching(W0_guess, T0_guess, airplane)
                W0 = airplane['W0']/1000  # em kN
                W0plot.append(W0)

                # salva no banco de dados
                results.append({
                    "Flap_type": flap_type,
                    "Sweep": sweep_deg,
                    "S_w": Sw,
                    "W0": W0
                })
            
        # Converte resultados em DataFrame
        df = pd.DataFrame(results)

        # Extrair sweeps únicos dessa config
        sweep_values = df[df["Flap_type"] == flap_type]["Sweep"].unique()

        # Definir colormap para diferenciar sweeps
        cmap = cm.plasma  
        norm = mcolors.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))

        # Criar figura + eixo
        fig, ax = plt.subplots(figsize=(10,8))

        # Plotar curvas
        for sweep_deg in sweep_values:
            df_sub = df[(df["Flap_type"] == flap_type) & (df["Sweep"] == sweep_deg)].sort_values("S_w")
            ax.plot(df_sub["S_w"], df_sub["W0"], color=cmap(norm(sweep_deg)))

        # Barra de cores em vez de legenda
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Enflechamento da asa (°)")

        # Configurações do gráfico
        ax.set_xlabel("Área da asa $S_w$ (m²)")
        ax.set_ylabel("W0 (kN)")
        ax.set_title(f"MTOW vs Área da asa para configuração de flap {flap_type}")
        ax.grid(True)
        ax.set_xlim(40, 160)
        ax.set_ylim(420, 570)

        figs.append(fig)

    plt.show()

    # Converte resultados em DataFrame
    df = pd.DataFrame(results)
    return df

def plot_W0_x_sweep(airplane, Swvec, sweep_wing_v, W0_guess, T0_guess):
    # Lista para armazenar resultados
    results = []

    # Loop sobre cada área de asa
    for Sw in Swvec:
        W0plot = []
        # Loop sobre cada enflechamento
        for sweep_wing in sweep_wing_v:
            airplane['sweep_w'] = sweep_wing
            airplane['S_w'] = Sw
            dt.geometry(airplane)

            dt.thrust_matching(W0_guess, T0_guess, airplane)
            W0 = airplane['W0']/1000  # em kN
            W0plot.append(W0)

            # salva no banco de dados
            results.append({
                "Sweep": sweep_wing*180/np.pi,
                "S_w": Sw,
                "W0": W0
            })
                

    # Converte resultados em DataFrame
    df = pd.DataFrame(results)

    # Extrai vetores
    Sw_values = df["S_w"].unique()
    
    # Define colormap
    cmap = cm.viridis  # pode trocar por 'plasma', 'coolwarm', etc
    norm = mcolors.Normalize(vmin=min(Sw_values), vmax=max(Sw_values))

    # Cria figura + eixo
    fig, ax = plt.subplots(figsize=(10,6))

    # Plota cada curva Sw com cor do colormap
    for Sw in Sw_values:
        df_sub = df[df["S_w"] == Sw].sort_values("Sweep")  # garantir ordem
        W0plot = df_sub["W0"].values
        sweep = df_sub["Sweep"].values
        ax.plot(sweep, W0plot, color=cmap(norm(Sw)))

    # Adiciona barra de cores no lugar da legenda
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Área da asa $S_w$ (m²)")

    ax.set_xlabel('Enflechamento (°)')
    ax.set_ylabel('W0 (kN)')
    ax.set_title('MTOW vs Enflechamento para diferentes Áreas de asa')
    ax.grid(True)
    
    plt.show()

    return df