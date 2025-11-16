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

def drag_polar(airplane, CL_cruise, num, save_path=None):
    """
    Plota as polares de arrasto para as configurações de cruise, takeoff e landing.
    
    Args:
        airplane (dict): Dicionário da aeronave.
        CL_cruise (float): CL de cruzeiro para marcar no gráfico.
        num (int): Número da aeronave para o título.
        save_path (str, optional): Caminho para salvar o gráfico (ex: "polar.pdf").
                                   Se None, apenas exibe com plt.show().
    """
    
    labels = ['Cruise', 'Takeoff', 'Landing']
    
    Mach_stall_takeoff, _ = get_Mach_stall(airplane, airplane['W0'], config='takeoff')
    Mach_stall_landing, _ = get_Mach_stall(airplane, 0.85*airplane['W0'], config='landing') 
    configs = [
        {'M': airplane['Mach_cruise'], 'H': airplane['altitude_cruise'], 'W': 0.95*airplane['W0'], 'config': 'clean'},
        {'M': 1.2*Mach_stall_takeoff, 'H': 0, 'W': airplane['W0'], 'config': 'takeoff'},
        {'M': 1.3*Mach_stall_landing, 'H': 0, 'W': 0.85*airplane['W0'],'config': 'landing'}
    ]
    
    colors = ['#0033A0', '#D40000', '#007A33'] # Cores (Azul, Vermelho, Verde)
    linestyles = ['-', '--', ':']
    
    # --- Início do Plot "Bonito" ---
    plt.figure(figsize=(10, 7)) # Tamanho maior
    
    for label, conf, color, ls in zip(labels, configs, colors, linestyles):
        CL_list = []
        CD_list = []
        # Usamos um range mais fino para uma curva suave
        for CL in np.arange(-0.5, 3.0, 0.01): 
            CD, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL,
                                        highlift_config=conf['config'],
                                        n_engines_failed=1 if conf['config']=='takeoff' else 0,
                                        lg_down=1 if conf['config']=='landing' else 0)
            CL_list.append(CL)
            CD_list.append(CD) # Não multiplicamos por 1e4 para manter o eixo em CD

        _, CLmax, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], 0.5, highlift_config=conf['config'])
        
        mask = np.array(CL_list) <= CLmax
        plt.plot(np.array(CD_list)[mask], np.array(CL_list)[mask], 
                 label=f"Config: {label}", color=color, linestyle=ls, linewidth=2)
        
        # Ponto de Cruzeiro
        if label == 'Cruise':
            CD_cruise_val, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CL_cruise)
            plt.plot(CD_cruise_val, CL_cruise, 's', color=color, markersize=8,
                     label=f"Ponto de Cruzeiro (CL={CL_cruise:.3f})")

        # Ponto de CLmax
        CD_clmax, _, _ = dt.aerodynamics(airplane, conf['M'], conf['H'], CLmax, highlift_config=conf['config'])
        plt.plot(CD_clmax, CLmax, 'o', color=color, markersize=8)
        plt.text(CD_clmax + 0.005, CLmax, f"CLmax = {CLmax:.2f}", 
                 color=color, fontsize=12, va='center')
            
    # --- Configurações do Gráfico "Bonito" ---
    plt.xlabel("Coeficiente de Arrasto (CD)", fontsize=14, fontweight='bold')
    plt.ylabel("Coeficiente de Sustentação (CL)", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title(f"Aeronave {num} - Polares de Arrasto (CL x CD)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # Começa o eixo Y no 0
    plt.xlim(left=0)   # Começa o eixo X no 0
    
    # --- LÓGICA PARA SALVAR EM PDF ---
    plt.tight_layout() # Ajusta para não cortar os labels
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"--- Gráfico das polares salvo em: {save_path} ---")
        plt.close() # Fecha a figura após salvar
    else:
        plt.show() # Mostra o gráfico se não for salvar
        

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