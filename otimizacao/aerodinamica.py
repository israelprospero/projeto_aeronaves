import sys
import os
from tabulate import tabulate 

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.designTool as dt
import modules.utils as m
import numpy as np
import pprint
import matplotlib.pyplot as plt

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi
a = 331.3  # m/s^2

def analise_aerodinamica(airplane, airplane_original=None, show_results=True):
    """
    Realiza a análise aerodinâmica completa e gera tabelas formatadas.
    """

    # ========================================================================
    # 1. ANÁLISE DA AERONAVE ATUAL (OTIMIZADA)
    # ========================================================================

    # ------ CL - Cruise ------ #
    M = airplane['Mach_cruise']
    H = airplane['altitude_cruise']
    V = M * m.get_a(H)
    
    MTOW = airplane['W0'] # N
    rho = dt.atmosphere(H)[2] 
    CL_cruise = 0.95 * MTOW / (0.5 * rho * V**2 * airplane['S_w']) # 95% of MTOW

    CD_cruise, _, dragDict = dt.aerodynamics(airplane, M, H, CL_cruise)
    LD_cruise = CL_cruise / CD_cruise

    # ------ CL - Landing ------ #
    MLW = airplane['MLW_frac']*airplane['W0'] 
    H_landing = 0 

    M_stall_landing, CLmax_landing = m.get_Mach_stall(airplane, MLW, 'landing', H_landing)

    M_landing = M_stall_landing * 1.3
    CL_landing = CLmax_landing / (1.3**2)

    CD_landing, _, dragDict_landing = dt.aerodynamics(airplane, M_landing, H_landing, 
                                                        CL_landing, 
                                                        n_engines_failed=0, highlift_config='landing', 
                                                        lg_down=1, h_ground=0, method=2)

    # ========================================================================
    # 2. PREPARAÇÃO DA TABELA 2: DRAG BREAKDOWN
    # ========================================================================
    # Extração dos componentes individuais
    cd_w = dragDict.get('CD0_w', 0)
    cd_f = dragDict.get('CD0_f', 0)
    cd_h = dragDict.get('CD0_h', 0)
    cd_v = dragDict.get('CD0_v', 0)
    cd_n = dragDict.get('CD0_n', 0)
    
    # Soma do parasita limpo
    cd_clean_sum = cd_w + cd_f + cd_h + cd_v + cd_n
    
    cd_ind = dragDict.get('CDind', 0)
    cd_wave = dragDict.get('CDwave', 0)
    cd_exc = dragDict.get('CD0_exc', 0)
    
    # Função auxiliar para formatar linha (Nome, Valor, %)
    def fmt_row(name, val, total):
        perc = (val / total) * 100
        return [name, f"{val:.5f}", f"{perc:.1f} %"]

    # Construção das linhas da tabela (Sem separadores manuais feios)
    tabela_2_data = [
        fmt_row("Asa (CD0_w)", cd_w, CD_cruise),
        fmt_row("Fuselagem (CD0_f)", cd_f, CD_cruise),
        fmt_row("Empenagem Horizontal (CD0_h)", cd_h, CD_cruise),
        fmt_row("Empenagem Vertical (CD0_v)", cd_v, CD_cruise),
        fmt_row("Naceles (CD0_n)", cd_n, CD_cruise),
        fmt_row(">> Arrasto Parasita (CD0_clean)", cd_clean_sum, CD_cruise), # Destaque visual
        fmt_row("Arrasto Induzido (CDind)", cd_ind, CD_cruise),
        fmt_row("Arrasto de Onda (CDwave)", cd_wave, CD_cruise),
        fmt_row("Arrasto de Excrescência (CD0_exc)", cd_exc, CD_cruise),
        ["ARRASTO TOTAL (CD_cruise)", f"{CD_cruise:.5f}", "100 %"],
        ["Eficiência (L/D)", f"{LD_cruise:.2f}", "—"]
    ]

    # ========================================================================
    # 3. PREPARAÇÃO DA TABELA 4: HIGH-LIFT PERFORMANCE
    # ========================================================================
    configs_to_test = [
        {'name': 'Limpa (Clean)', 'cfg': 'clean', 'eng': 0, 'lg': 0},
        {'name': 'Decolagem (Takeoff)', 'cfg': 'takeoff', 'eng': 1, 'lg': 0},
        {'name': 'Pouso (Landing)', 'cfg': 'landing', 'eng': 0, 'lg': 1}
    ]

    table_4_data = []

    for c in configs_to_test:
        if c['cfg'] == 'landing':
            W_ref = MLW
        else:
            W_ref = MTOW
            
        M_stall, CLmax_val = m.get_Mach_stall(airplane, W_ref, c['cfg'], 0)
        
        CD_at_max, _, dDict = dt.aerodynamics(airplane, M_stall, 0, CLmax_val, 
                                              highlift_config=c['cfg'], 
                                              n_engines_failed=c['eng'], 
                                              lg_down=c['lg'])
        
        LD_at_max = CLmax_val / CD_at_max
        
        dCL_flap = dDict.get('deltaCLmax_flap', 0.0)
        dCL_slat = dDict.get('deltaCLmax_slat', 0.0)
        
        table_4_data.append([
            c['name'], 
            f"{CLmax_val:.3f}", 
            f"{CD_at_max:.4f}", 
            f"{LD_at_max:.2f}", 
            f"{dCL_flap:.3f}", 
            f"{dCL_slat:.3f}"
        ])

    # ========================================================================
    # 4. PREPARAÇÃO DA COMPARAÇÃO (SE HOUVER AERONAVE ORIGINAL)
    # ========================================================================
    tabela_comp = None 
    label_comp_plot = None

    if airplane_original is not None:
        dt.analyze(airplane_original, print_log=False, plot=False)
        
        M_ref = airplane_original['Mach_cruise']
        H_ref = airplane_original['altitude_cruise']
        V_ref = M_ref * m.get_a(H_ref)
        rho_ref = dt.atmosphere(H_ref)[2]
        CL_cruise_ref = 0.95 * airplane_original['W0'] / (0.5 * rho_ref * V_ref**2 * airplane_original['S_w'])
        CD_cruise_ref, _, _ = dt.aerodynamics(airplane_original, M_ref, H_ref, CL_cruise_ref)
        LD_cruise_ref = CL_cruise_ref / CD_cruise_ref

        def calc_diff(opt, ref):
            if ref == 0: return 0.0
            return ((opt - ref) / ref) * 100

        opt_mtow = airplane['W0']
        ref_mtow = airplane_original['W0']
        
        opt_sw = airplane['S_w']
        ref_sw = airplane_original['S_w']

        opt_ar = airplane['AR_w']
        ref_ar = airplane_original['AR_w']

        tabela_comp = [
            ["Parâmetro", "Otimizada", "Original", "Diferença (%)"],
            ["MTOW [N]", f"{opt_mtow:.0f}", f"{ref_mtow:.0f}", f"{calc_diff(opt_mtow, ref_mtow):+.2f}%"],
            ["Área Alar [m2]", f"{opt_sw:.1f}", f"{ref_sw:.1f}", f"{calc_diff(opt_sw, ref_sw):+.2f}%"],
            ["Alongamento (AR)", f"{opt_ar:.2f}", f"{ref_ar:.2f}", f"{calc_diff(opt_ar, ref_ar):+.2f}%"],
            ["CL Cruzeiro", f"{CL_cruise:.3f}", f"{CL_cruise_ref:.3f}", f"{calc_diff(CL_cruise, CL_cruise_ref):+.2f}%"],
            ["CD Cruzeiro", f"{CD_cruise:.5f}", f"{CD_cruise_ref:.5f}", f"{calc_diff(CD_cruise, CD_cruise_ref):+.2f}%"],
            ["L/D Cruzeiro", f"{LD_cruise:.2f}", f"{LD_cruise_ref:.2f}", f"{calc_diff(LD_cruise, LD_cruise_ref):+.2f}%"]
        ]
        
        label_comp_plot = 'Original'


    # ========================================================================
    # 5. IMPRESSÃO DOS RESULTADOS E GRÁFICOS
    # ========================================================================

    if show_results:
        print('\n' + '='*60)
        print(f" RELATÓRIO AERODINÂMICO: {airplane['name']}")
        print('='*60 + '\n')

        # --- TABELA 2: DRAG BREAKDOWN ---
        # AGORA COM FANCY_GRID
        print(f"### Tabela 2: Drag breakdown em condição de cruzeiro (M={M}, CL={CL_cruise:.3f}) ###")
        print(tabulate(tabela_2_data, headers=["Componente de Arrasto", "Coeficiente (CD)", "Contribuição (%)"], tablefmt="fancy_grid"))
        print("\n")

        # --- TABELA 3: DISPOSITIVOS ---
        print("### Tabela 3: Definição dos Dispositivos Hipersustentadores ###")
        tabela_3 = [
            ["Flap", airplane['flap_type']],
            ["Slat", str(airplane['slat_type'])]
        ]
        print(tabulate(tabela_3, headers=["Componente", "Tipo Utilizado"], tablefmt="fancy_grid"))
        print("\n")

        # --- TABELA 4: PERFORMANCE ---
        print("### Tabela 4: Desempenho Aerodinâmico em Diferentes Configurações ###")
        headers_t4 = ["Configuração", "CLmax", "CD (@CLmax)", "L/D (@CLmax)", "dCLmax Flap", "dCLmax Slat"]
        print(tabulate(table_4_data, headers=headers_t4, tablefmt="fancy_grid"))
        print("\n")

        # --- TABELA COMPARATIVA ---
        if tabela_comp:
            print("### Comparativo Aerodinâmico e de Peso (Otimizada vs Original) ###")
            print(tabulate(tabela_comp, headers="firstrow", tablefmt="fancy_grid"))
            print("\n")

        # --- PLOTS ---
        
        # 1. Plot CD x Mach
        M_range = np.arange(0.6, 0.9, 0.001)
        # m.plot_CD_x_M(M_range, H, CL_cruise, airplane, 0.95*airplane['W0'], '1') 

        # 2. Plot Drag Polar
        nome_do_pdf = "drag_polar.pdf"
        path_save = os.path.join(script_dir, nome_do_pdf)
        
        print(f"Gerando gráfico polar em: {path_save}")
        
        m.drag_polar(airplane, 
             CL_cruise, 
             1,
             save_path=path_save, 
             airplane_comp=airplane_original,
             label_comp=label_comp_plot if label_comp_plot else "Comparação")

    # Atualiza chaves do dicionário principal para retorno
    airplane['aero_CL_cruise'] = CL_cruise
    airplane['aero_CD_cruise'] = CD_cruise
    airplane['aero_CL_landing'] = CL_landing
    airplane['aero_CD_landing'] = CD_landing
    airplane['aero_CLmax_landing'] = CLmax_landing
    airplane['aero_LD_cruise'] = LD_cruise
    
    return airplane