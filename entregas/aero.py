import numpy as np
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import modules.designTool as dt 
import modules.utils as ut

print("--- INICIANDO ANÁLISE DA AERONAVE ---")

airplane = dt.standard_airplane(name='fokker100')
print(f"Analisando o avião: {airplane['name']}")

airplane = dt.analyze(airplane, print_log=False, plot=False)

W0 = airplane['W0']
W_cruise = W0 * airplane['Mf_engine_start'] * airplane['Mf_taxi'] * \
           airplane['Mf_takeoff'] * airplane['Mf_climb']
rho_cruise = dt.atmosphere(airplane['altitude_cruise'])[2]
a_cruise = ut.get_a(airplane['altitude_cruise'])
V_cruise = airplane['Mach_cruise'] * a_cruise
S_w = airplane['S_w']
CL_cruise = (2 * W_cruise) / (rho_cruise * S_w * V_cruise**2)
CD_cruise, _, dragDict_cruise = dt.aerodynamics(airplane, 
                                                 Mach=airplane['Mach_cruise'], 
                                                 altitude=airplane['altitude_cruise'], 
                                                 CL=CL_cruise,
                                                 highlift_config='clean')
LD_cruise = CL_cruise / CD_cruise

# Configuração LIMPA (Clean)
Mach_stall_clean, CLmax_clean = ut.get_Mach_stall(airplane, airplane['W0'] * airplane['MLW_frac'], config='clean')
CD_at_CLmax_clean, _, dragDict_clean = dt.aerodynamics(airplane, 
                                                  Mach=Mach_stall_clean, 
                                                  altitude=0, 
                                                  CL=CLmax_clean,
                                                  highlift_config='clean', 
                                                  lg_down=0) 
LD_at_CLmax_clean = CLmax_clean / CD_at_CLmax_clean

# Configuração DECOLAGEM (Takeoff)
Mach_stall_takeoff, CLmax_TO = ut.get_Mach_stall(airplane, airplane['W0'], config='takeoff')
CD_at_CLmax_TO, _, dragDict_TO = dt.aerodynamics(airplane, 
                                      Mach=Mach_stall_takeoff, 
                                      altitude=0, 
                                      CL=CLmax_TO,
                                      highlift_config='takeoff', 
                                      lg_down=1) 
LD_at_CLmax_TO = CLmax_TO / CD_at_CLmax_TO

# Configuração POUSO (Landing)
Mach_stall_land, CLmax_LD = ut.get_Mach_stall(airplane, airplane['W0'] * airplane['MLW_frac'], config='landing')
CD_at_CLmax_LD, _, dragDict_land = dt.aerodynamics(airplane, 
                                          Mach=Mach_stall_land, 
                                          altitude=0, 
                                          CL=CLmax_LD,
                                          highlift_config='landing', 
                                          lg_down=1) 
LD_at_CLmax_LD = CLmax_LD / CD_at_CLmax_LD

nome_do_pdf = "drag_polar_definitivo.pdf"
path_save = os.path.join(script_dir, nome_do_pdf)

print(f"\n--- GERANDO GRÁFICO DAS POLARES---")
print(f"Caminho de salvamento: {path_save}")

ut.drag_polar(airplane, CL_cruise, num=1, save_path=path_save)

#####################################################################
###           SUMÁRIO DE DADOS PARA COPIAR NO LATEX               ###
#####################################################################

print("\n\n--- DADOS PARA TABELA LATEX (COPIE DAQUI) ---")

# 1. PARÂMETROS GERAIS (Para Tabela 1)
print("\n### 1. Parâmetros Gerais ###")
print(f"W0 (MTOW) [N]: {airplane['W0']:.0f}")
print(f"T0 (Thrust) [N]: {airplane['T0']:.0f}")
print(f"T0/W0: {airplane['T0']/airplane['W0']:.3f}")
print(f"W0/S [N/m2]: {airplane['W0']/airplane['S_w']:.1f}")

# 2. BREAKDOWN DE ARRASTO (Para Tabela 2)
print("\n### 2. Breakdown de Arrasto (Cruzeiro) ###")
print(f"CL_cruise: {CL_cruise:.4f}")
print(f"CD_cruise (total): {CD_cruise:.5f}")
print(f"(L/D)_cruise: {LD_cruise:.2f}")
print(f"--- Componentes (CD * 1e4) ---")
print(f"CD0_w (Asa): {dragDict_cruise['CD0_w']*1e4:.2f}")
print(f"CD0_f (Fuselagem): {dragDict_cruise['CD0_f']*1e4:.2f}")
print(f"CD0_h (Emp. Horiz.): {dragDict_cruise['CD0_h']*1e4:.2f}")
print(f"CD0_v (Emp. Vert.): {dragDict_cruise['CD0_v']*1e4:.2f}")
print(f"CD0_n (Naceles): {dragDict_cruise['CD0_n']*1e4:.2f}")
print(f"CD0_clean (Soma): {dragDict_cruise['CD0']*1e4:.2f}")
print(f"CD_ind (Induzido): {dragDict_cruise['CDind']*1e4:.2f}")
print(f"CD_wave (Onda): {dragDict_cruise['CDwave']*1e4:.2f}")
print(f"CD0_exc (Excresc.): {dragDict_cruise['CD0_exc']*1e4:.2f}")

# 3. TIPO DE DISPOSITIVO (Para Tabela 3)
print("\n### 3. Tipo de Dispositivo ###")
print(f"Flap Type: {airplane['flap_type']}")
print(f"Slat Type: {airplane['slat_type']}")

# 4. TABELA DE HIGH-LIFT COMPLETA (Para Tabela 4)
print("\n### 4. Tabela de High-Lift Completa ###")
print("--- Config: Limpa (Clean) ---")
print(f"CLmax (Clean): {CLmax_clean:.4f}")
print(f"CD (@CLmax Clean): {CD_at_CLmax_clean:.5f}")
print(f"L/D (@CLmax Clean): {LD_at_CLmax_clean:.2f}")
print(f"Delta_CLmax_Flap (Clean): {dragDict_clean['deltaCLmax_flap']:.4f}")
print(f"Delta_CLmax_Slat (Clean): {dragDict_clean['deltaCLmax_slat']:.4f}")

print("\n--- Config: Decolagem (Takeoff) ---")
print(f"CLmax (Takeoff): {CLmax_TO:.4f}")
print(f"CD (@CLmax Takeoff): {CD_at_CLmax_TO:.5f}")
print(f"L/D (@CLmax Takeoff): {LD_at_CLmax_TO:.2f}")
print(f"Delta_CLmax_Flap (Takeoff): {dragDict_TO['deltaCLmax_flap']:.4f}")
print(f"Delta_CLmax_Slat (Takeoff): {dragDict_TO['deltaCLmax_slat']:.4f}")

print("\n--- Config: Pouso (Landing) ---")
print(f"CLmax (Landing): {CLmax_LD:.4f}")
print(f"CD (@CLmax Landing): {CD_at_CLmax_LD:.5f}")
print(f"L/D (@CLmax Landing): {LD_at_CLmax_LD:.2f}")
print(f"Delta_CLmax_Flap (Landing): {dragDict_land['deltaCLmax_flap']:.4f}")
print(f"Delta_CLmax_Slat (Landing): {dragDict_land['deltaCLmax_slat']:.4f}")

print("\n--- ANÁLISE CONCLUÍDA ---")
print(f"Verifique o arquivo {path_save} na pasta.")