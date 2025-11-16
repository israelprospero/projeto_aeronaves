"""
MDO Analysis Suite
==================

Conjunto de ferramentas para análise de aeronaves integrado com o designTool.py.
"""

# === IMPORTAÇÕES ===
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
import copy
import warnings
import os
import sys
import argparse
from tabulate import tabulate

# Configuração de Caminho para Importação
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Tenta importar o designTool
try:
    import modules.designTool as dt
except ImportError:
    try:
        import designTool as dt
    except ImportError:
        print("AVISO: Não foi possível importar 'designTool'. Verifique os caminhos.")
        class StubDesignTool:
            def standard_airplane(self, name): return {}
            def analyze(self, airplane, print_log=False, plot=False): pass
        dt = StubDesignTool()

# ================================================================
# === DEFINIÇÕES MESTRAS (MASTER DEFINITIONS) ===
# ================================================================

def get_perc_var(val):
    perc = 0.1
    if val > 0:
        return val - perc * val, val + perc * val
    else:
        return val + perc * val, val - perc * val

# Tenta carregar a aeronave para definir os bounds dinâmicos
try:
    airplane_ref = dt.standard_airplane('my_airplane_1')
except:
    airplane_ref = {} 

default_airplane_params = {
    'xr_w': 10.0, 'zr_w': 0.0, 'tcr_w': 0.12, 'tct_w': 0.12,
    'zr_h': 0.0, 'tcr_h': 0.1, 'tct_h': 0.1,
    'zr_v': 0.0, 'x_n': 8.0, 'y_n': 2.0, 'z_n': -1.0,
    'x_tailstrike': 20.0, 'z_tailstrike': 1.0,
    'c_tank_c_w': 0.5, 'b_tank_b_w_end': 0.8,
    'c_flap_c_wing': 0.2, 'b_flap_b_wing': 0.6,
    'c_ail_c_wing': 0.2, 'b_ail_b_wing': 0.3
}
if not airplane_ref:
    airplane_ref = default_airplane_params

MASTER_DOE_BOUNDS = {
    'S_w':              [70, 105],
    'AR_w':             [7, 11],
    'taper_w':          [0.2, 0.4],
    'sweep_w':          [20*np.pi/180, 30*np.pi/180],
    'dihedral_w':       [2*np.pi/180, 6.5*np.pi/180],
    'xr_w':             list(get_perc_var(airplane_ref.get('xr_w', default_airplane_params['xr_w']))),
    'zr_w':             list(get_perc_var(airplane_ref.get('zr_w', default_airplane_params['zr_w']))),
    'tcr_w':            list(get_perc_var(airplane_ref.get('tcr_w', default_airplane_params['tcr_w']))),
    'tct_w':            list(get_perc_var(airplane_ref.get('tct_w', default_airplane_params['tct_w']))),
    'Cht':              [0.9, 1.15],
    'Lc_h':             [3, 5.5],
    'AR_h':             [3.5, 5.5],
    'taper_h':          [0.3, 0.5],
    'sweep_h':          [23*np.pi/180, 35*np.pi/180],
    'dihedral_h':       [3*np.pi/180, 10*np.pi/180],
    'zr_h':             [0.6, 0.9],
    'tcr_h':            [0.05, 0.15],
    'tct_h':            [0.05, 0.15],
    'Cvt':              [0.06, 0.11],
    'Lb_v':             [0.35, 0.65],
    'AR_v':             [1, 2],
    'taper_v':          [0.3, 0.6],
    'sweep_v':          [30*np.pi/180, 50*np.pi/180],
    'zr_v':             list(get_perc_var(airplane_ref.get('zr_v', default_airplane_params['zr_v']))),
    'x_n':              [airplane_ref.get('xr_w', default_airplane_params['xr_w']) - 3, airplane_ref.get('xr_w', default_airplane_params['xr_w']) + 4],
    'y_n':              list(get_perc_var(airplane_ref.get('y_n', default_airplane_params['y_n']))),
    'z_n':              list(get_perc_var(airplane_ref.get('z_n', default_airplane_params['z_n']))),
    'L_n':              [3, 4.5],
    'D_n':              [1.5, 2.3],
    'x_nlg':            [2.5, 5.5],
    'x_mlg':            [airplane_ref.get('xr_w', default_airplane_params['xr_w']) + 1.7, airplane_ref.get('xr_w', default_airplane_params['xr_w']) + 3.9],
    'y_mlg':            [2, 4],
    'z_lg':             [-4, -2],
    'x_tailstrike':     list(get_perc_var(airplane_ref.get('x_tailstrike', default_airplane_params['x_tailstrike']))),
    'z_tailstrike':     list(get_perc_var(airplane_ref.get('z_tailstrike', default_airplane_params['z_tailstrike']))),
    'c_tank_c_w':       list(get_perc_var(airplane_ref.get('c_tank_c_w', default_airplane_params['c_tank_c_w']))),
    'b_tank_b_w_end':   list(get_perc_var(airplane_ref.get('b_tank_b_w_end', default_airplane_params['b_tank_b_w_end']))),
    'c_flap_c_wing':    list(get_perc_var(airplane_ref.get('c_flap_c_wing', default_airplane_params['c_flap_c_wing']))),
    'b_flap_b_wing':    list(get_perc_var(airplane_ref.get('b_flap_b_wing', default_airplane_params['b_flap_b_wing']))),
    'c_ail_c_wing':     list(get_perc_var(airplane_ref.get('c_ail_c_wing', default_airplane_params['c_ail_c_wing']))),
    'b_ail_b_wing':     list(get_perc_var(airplane_ref.get('b_ail_b_wing', default_airplane_params['b_ail_b_wing'])))
}

MASTER_OAT_INPUTS = list(MASTER_DOE_BOUNDS.keys())

MASTER_OUTPUTS = [
    # 'W0', 'W_empty', 'W_fuel', 'T0', 'DOC', 'deltaS_wlan', 'tank_excess',
    # 'CLv', 'SM_fwd', 'SM_aft', 'alpha_tipback', 'alpha_tailstrike', 'phi_overturn'
    'frac_nlg_fwd', 'frac_nlg_aft'
]

MASTER_PAIRGRID_SUBSET = [
    # 'W0', 'W_empty', 'W_fuel', 'T0', 'DOC', 'deltaS_wlan', 'tank_excess',
    # 'CLv', 'SM_fwd', 'SM_aft', 'alpha_tipback', 'alpha_tailstrike', 'phi_overturn'
    'frac_nlg_fwd', 'frac_nlg_aft'
]

# =============================================================================
# === FUNÇÃO 1: ANÁLISE PARAMÉTRICA (OAT) ===
# =============================================================================
def analise_parametrica(baseline_airplane, 
                        variaveis_entrada, 
                        variaveis_saida, 
                        intervalo_percentual=(0.8, 1.2), 
                        num_passos=11,
                        output_dir="."):
    
    # Validação de entrada
    if num_passos < 3 or num_passos % 2 == 0:
        print(f"AVISO: 'num_passos' deve ser ímpar e >= 3. Ajustando para {max(3, num_passos + (1 - num_passos % 2))}.")
        num_passos = max(3, num_passos + (1 - num_passos % 2))

    print("--- Iniciando Análise Paramétrica (OAT) Robusta ---")
    
    # Criação de diretórios de saída
    output_plots_dir_grid = os.path.join(output_dir, "oat_plots_grid")
    output_plots_dir_single = os.path.join(output_dir, "oat_plots_single")
    if not os.path.exists(output_plots_dir_grid):
        os.makedirs(output_plots_dir_grid)
    if not os.path.exists(output_plots_dir_single):
        os.makedirs(output_plots_dir_single)

    csv_filename = os.path.join(output_dir, "oat_analise_detalhada.csv")

    # --- Etapa 0 - Calcular o Ponto de Referência (Baseline) ---
    print("Calculando baseline de referência (100%)...")
    baseline_analisada = copy.deepcopy(baseline_airplane)
    y_baseline_valores = {}
    try:
        dt.analyze(baseline_analisada, print_log=False, plot=False)
        for var_out in variaveis_saida:
            y_baseline_valores[var_out] = baseline_analisada.get(var_out, None)
        print("Baseline calculada com sucesso.")
    except Exception as e:
        print(f"ERRO FATAL: A análise da aeronave de baseline falhou. Erro: {e}")
        return None, None
    
    resultados_grafico = {}
    dados_para_csv = []

    for var_in in variaveis_entrada:
        print(f"Analisando variável de entrada: {var_in}")
        
        try:
            valor_base = baseline_airplane[var_in]
        except KeyError:
            print(f"AVISO: Variável de entrada '{var_in}' não encontrada no baseline. Pulando.")
            continue
            
        vetor_absoluto = np.linspace(valor_base * intervalo_percentual[0], 
                                     valor_base * intervalo_percentual[1], 
                                     num_passos)
        vetor_normalizado = (vetor_absoluto / valor_base) * 100
        
        dados_saida_var = {var_out: [] for var_out in variaveis_saida}
        
        for i, valor_perturbado in enumerate(vetor_absoluto):
            aeronave_temp = copy.deepcopy(baseline_airplane)
            aeronave_temp[var_in] = valor_perturbado
            linha_csv = {'variavel_entrada': var_in, 'percentual_entrada': vetor_normalizado[i], 'valor_entrada': valor_perturbado}
            
            try:
                dt.analyze(aeronave_temp, print_log=False, plot=False)
                for var_out in variaveis_saida:
                    valor_saida = aeronave_temp.get(var_out, None)
                    dados_saida_var[var_out].append(valor_saida)
                    linha_csv[var_out] = valor_saida
            except Exception as e:
                for var_out in variaveis_saida:
                    dados_saida_var[var_out].append(None)
                    linha_csv[var_out] = None
            
            dados_para_csv.append(linha_csv)

        resultados_grafico[var_in] = {
            'vetor_normalizado_x': vetor_normalizado, 'vetor_absoluto_x': vetor_absoluto, 'dados_saida': dados_saida_var
        }

    print("--- Análise concluída. Gerando gráficos. ---")

    # === GRÁFICO ÚNICO (ESPAGUETE) MELHORADO ===
    num_vars_entrada = len(variaveis_entrada)
    if num_vars_entrada > 0:
        if num_vars_entrada <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, num_vars_entrada))
        else:
            # CORREÇÃO AQUI: Usando nipy_spectral que existe no matplotlib.cm
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_vars_entrada))
    else:
        colors = []

    color_map = {var_in: colors[i] for i, var_in in enumerate(variaveis_entrada)}

    for var_out in variaveis_saida:
        plt.figure(figsize=(12, 8))

        for var_in in variaveis_entrada:
            if var_in in resultados_grafico:
                eixo_x = resultados_grafico[var_in]['vetor_normalizado_x']
                eixo_y = resultados_grafico[var_in]['dados_saida'][var_out]
                
                plt.plot(eixo_x, eixo_y, 'o-', 
                         label=f"{var_in}", 
                         color=color_map.get(var_in, 'gray'),
                         markersize=4, 
                         linewidth=1.5)
        
        plt.title(f"Análise de Sensibilidade (OAT) para: {var_out}", fontsize=18, weight='bold')
        plt.xlabel("Variação % em relação ao valor de referência", fontsize=14)
        plt.ylabel(f"Valor de Saída: {var_out}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axvline(x=100, color='red', linestyle=':', linewidth=2, label='Baseline (100%)')
        plt.tick_params(axis='both', which='major', labelsize=10)

        if num_vars_entrada > 0:
            n_legend_cols = math.ceil(len(variaveis_entrada) / 5) 
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=n_legend_cols, 
                       fontsize='small', title="Variáveis de Entrada", title_fontsize='medium')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plot_filename_single = os.path.join(output_plots_dir_single, f"oat_plot_single__{var_out}.png")
        try:
            plt.savefig(plot_filename_single, dpi=200, bbox_inches='tight')
        except Exception as e:
            print(f"ERRO ao salvar gráfico 'Espaguete' {plot_filename_single}: {e}")
        plt.close()


    # === GRÁFICO GRID (SMALL MULTIPLES) ===
    sns.set(style="whitegrid")
    
    for var_out in variaveis_saida:
        num_vars = len(resultados_grafico)
        if num_vars == 0: continue

        n_cols = 5
        n_rows = math.ceil(num_vars / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), sharey=True)
        axes_flat = axes.flatten()
        
        fig.suptitle(f"Sensibilidade (OAT): Impacto em {var_out} (Grid)", fontsize=20, weight='bold', y=1.01)
        
        idx_plot = 0
        for var_in, res in resultados_grafico.items():
            ax = axes_flat[idx_plot]
            x_vals = res['vetor_normalizado_x']
            y_vals = res['dados_saida'][var_out]
            
            ax.plot(x_vals, y_vals, 'o-', color='#2b7bba', markersize=4, linewidth=2)
            ax.axvline(x=100, color='#e74c3c', linestyle=':', linewidth=1.5, label='Ref')
            
            ax.set_title(var_in, fontsize=11, weight='bold')
            
            if idx_plot >= (n_rows - 1) * n_cols: 
                ax.set_xlabel("% Ref")
            
            idx_plot += 1
            
        for i in range(idx_plot, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        plot_filename_grid = os.path.join(output_plots_dir_grid, f"oat_GRID__{var_out}.png")
        try:
            plt.savefig(plot_filename_grid, dpi=200, bbox_inches='tight')
        except Exception as e:
            print(f"ERRO ao salvar gráfico GRID {plot_filename_grid}: {e}")
        plt.close()

    print(f"--- Gráficos salvos em '{output_dir}'. ---")
    
    try:
        print(f"\nSalvando CSV em '{csv_filename}'...")
        df = pd.DataFrame(dados_para_csv)
        colunas_ordem = ['variavel_entrada', 'percentual_entrada', 'valor_entrada'] + variaveis_saida
        df = df[colunas_ordem]
        df.to_csv(csv_filename, index=False, float_format='%.5e')
    except Exception as e:
        print(f"ERRO ao salvar CSV: {e}")

    print("\n--- Tabela de Análise de Sensibilidade (no Ponto de Referência) ---")
    tabela_sensibilidade = []
    headers = ["Variável Saída (y)", "Variável Entrada (x)", "Sensib. Absoluta (dy/dx)", "Sensib. Relativa (dy/y*)/(dx/x*)"]
    idx_base = num_passos // 2
    idx_pre, idx_post = idx_base - 1, idx_base + 1
    
    for var_out in variaveis_saida:
        y_star = y_baseline_valores.get(var_out)
        if not y_star: continue
            
        for var_in in variaveis_entrada:
            if var_in not in resultados_grafico: continue
            x_star = baseline_airplane.get(var_in)
            res = resultados_grafico[var_in]
            try:
                x_pre, x_post = res['vetor_absoluto_x'][idx_pre], res['vetor_absoluto_x'][idx_post]
                y_pre, y_post = res['dados_saida'][var_out][idx_pre], res['dados_saida'][var_out][idx_post]
                if y_pre is None or y_post is None: continue
                
                delta_y = y_post - y_pre
                delta_x = x_post - x_pre
                if delta_x == 0: continue

                abs_sens = delta_y / delta_x
                rel_sens = abs_sens * (x_star / y_star) if x_star and y_star else 0
                tabela_sensibilidade.append([var_out, var_in, abs_sens, rel_sens])
            except:
                pass
    
    try:
        tabela_sensibilidade.sort(key=lambda x: abs(float(x[3])) if isinstance(x[3], (int, float)) else 0, reverse=True)
    except:
        pass

    print(tabulate(tabela_sensibilidade, headers=headers, floatfmt=".4f", tablefmt="grid"))
    print("--- Fim da Análise Paramétrica ---")

    return resultados_grafico, df

# =============================================================================
# === FUNÇÃO 2: DESIGN OF EXPERIMENTS (DOE) ===
# =============================================================================

def corrdot(*args, **kwargs):
    x_data, y_data = args[0], args[1]
    valid_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    if len(valid_data) < 2: corr_r = np.nan
    else: corr_r = valid_data['x'].corr(valid_data['y'], 'pearson')
    
    font_size = abs(corr_r) * 40 + 5
    marker_size = abs(corr_r) * 10000
    
    ax = plt.gca(); ax.set_axis_off()
    if pd.isna(corr_r): return

    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm", vmin=-1, vmax=1, transform=ax.transAxes)
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax.annotate(corr_text, [.5, .5], xycoords="axes fraction", ha='center', va='center', fontsize=font_size)

def _gerar_pairgrid(df_plot, title, filename, salvar_graficos, show_plot=True):
    try:
        n_plot_vars = len(df_plot.columns)
        altura_subplot = max(1.5, min(3.0, 15 / n_plot_vars))
        
        sns.set(style='white', font_scale=1.1)
        fig = sns.PairGrid(df_plot, diag_sharey=False, height=altura_subplot)
        fig.map_diag(sns.histplot, kde=True, color='#4a90e2')
        fig.map_lower(sns.regplot, lowess=True, scatter_kws={'alpha': 0.3, 's': 15, 'color': '#2c3e50'}, line_kws={'color': '#e74c3c', 'lw': 2})
        fig.map_upper(corrdot)
        fig.fig.suptitle(title, y=1.02, fontsize=16, weight='bold')
        plt.tight_layout()
        
        if salvar_graficos:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   Gráfico de matriz salvo em '{filename}'")
            except Exception as e:
                print(f"   ERRO ao salvar gráfico: {e}")
        if show_plot: plt.show()
        else: plt.close(fig.fig)
    except Exception as e:
        print(f"   ERRO ao gerar PairGrid: {e}")
        plt.close()

def executar_doe_analysis(
        baseline_airplane,
        variaveis_entrada_bounds,
        variaveis_saida,
        n_samples=100,
        sampler_method=LHS(),
        plot_style='pairgrid',
        plot_vars_subset=None,
        output_dir=".",
        matrix_chunk_X=3,
        matrix_chunk_Y=2,
        salvar_graficos=True):
    
    print(f"--- Iniciando DOE com {n_samples} amostras ---")
    
    csv_filename = os.path.join(output_dir, f"doe_results_{plot_style}_{n_samples}s.csv")
    individual_plots_dir = os.path.join(output_dir, "doe_plots_individuais")
    matrix_plots_dir = os.path.join(output_dir, "doe_plots_matrizes")
    pairgrid_filename = os.path.join(output_dir, "doe_correlation_plot.png")

    if salvar_graficos and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Criado diretório de saída: {output_dir}")

    variaveis_entrada_nomes = list(variaveis_entrada_bounds.keys())
    if not variaveis_entrada_nomes or not variaveis_saida:
        print("ERRO: Entradas ou Saídas não definidas.")
        return None
        
    n_var = len(variaveis_entrada_nomes)
    lb = [variaveis_entrada_bounds[key][0] for key in variaveis_entrada_nomes]
    ub = [variaveis_entrada_bounds[key][1] for key in variaveis_entrada_nomes]

    problem = Problem(n_var=n_var, xl=lb, xu=ub)
    print(f"Gerando amostras com {sampler_method.__class__.__name__}...")
    X_samples = sampler_method(problem, n_samples).get("X")

    all_results_data = []
    print(f"Executando {n_samples} amostras...")
    
    for i in range(n_samples):
        x_sample = X_samples[i, :]
        aeronave_temp = copy.deepcopy(baseline_airplane)
        input_dict = {}
        
        for j, var_in in enumerate(variaveis_entrada_nomes):
            aeronave_temp[var_in] = x_sample[j]
            input_dict[var_in] = x_sample[j]
            
        try:
            dt.analyze(aeronave_temp, print_log=False, plot=False)
            output_dict = {var_out: aeronave_temp.get(var_out, np.nan) for var_out in variaveis_saida}
        except:
            output_dict = {var_out: np.nan for var_out in variaveis_saida}
        
        all_results_data.append({**input_dict, **output_dict})

    print("--- Análise DOE concluída ---")
    df = pd.DataFrame(all_results_data)
    
    if csv_filename and salvar_graficos:
        try:
            df.to_csv(csv_filename, index=False, float_format='%.5e')
            print(f"Resultados salvos em '{csv_filename}'")
        except Exception as e:
            print(f"ERRO CSV: {e}")

    if not salvar_graficos: return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if plot_style == 'individual' or plot_style == 'full_report':
            print("Gerando gráficos individuais...")
            if not os.path.exists(individual_plots_dir): os.makedirs(individual_plots_dir)
            
            for var_out in variaveis_saida:
                for var_in in variaveis_entrada_nomes:
                    if df[var_in].isnull().all() or df[var_out].isnull().all(): continue
                    
                    plt.figure(figsize=(6, 4))
                    sns.regplot(data=df, x=var_in, y=var_out, scatter_kws={'alpha': 0.5, 'color': '#34495e'}, line_kws={'color': '#e67e22', 'lw': 2})
                    plt.title(f"{var_out} vs. {var_in}")
                    plt.grid(True, linestyle='--', alpha=0.5)
                    
                    fname = os.path.join(individual_plots_dir, f"plot__{var_out}_vs_{var_in}.png")
                    plt.savefig(fname, dpi=100, bbox_inches='tight')
                    plt.close()
            print("Gráficos individuais salvos.")

        if plot_style == 'pairgrid' or plot_style == 'full_report':
            print("Gerando PairGrid...")
            if plot_vars_subset:
                cols = [v for v in plot_vars_subset if v in df.columns]
                if cols: _gerar_pairgrid(df[cols], "Matriz de Correlação DOE", pairgrid_filename, salvar_graficos, show_plot=False)
            else:
                _gerar_pairgrid(df, "Matriz de Correlação DOE", pairgrid_filename, salvar_graficos, show_plot=False)

        if plot_style == 'matrix_combinations' or plot_style == 'full_report':
            print("Gerando matrizes combinadas...")
            if not os.path.exists(matrix_plots_dir): os.makedirs(matrix_plots_dir)
            
            input_chunks = [variaveis_entrada_nomes[i:i + matrix_chunk_X] for i in range(0, len(variaveis_entrada_nomes), matrix_chunk_X)]
            output_chunks = [variaveis_saida[i:i + matrix_chunk_Y] for i in range(0, len(variaveis_saida), matrix_chunk_Y)]
            
            count = 0
            for i, in_chunk in enumerate(input_chunks):
                for j, out_chunk in enumerate(output_chunks):
                    df_sub = df[in_chunk + out_chunk]
                    fname = os.path.join(matrix_plots_dir, f"matriz_X{i+1}_Y{j+1}.png")
                    _gerar_pairgrid(df_sub, f"Matriz X{i+1} - Y{j+1}", fname, salvar_graficos, show_plot=False)
                    count += 1
            print("Matrizes combinadas salvas.")

    return df

# =============================================================================
# === CLI ===
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MDO Analysis Suite", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-dir', '--output_dir', type=str, default="mdo_analysis_output", help="Output directory")
    parser.add_argument('-b', '--baseline', type=str, default='fokker100', help="Baseline airplane name")

    subparsers = parser.add_subparsers(dest='analysis_type', required=True)

    # DOE
    p_doe = subparsers.add_parser('doe')
    p_doe.add_argument('-n', '--n_samples', type=int, default=100)
    p_doe.add_argument('-p', '--plot_style', type=str, default='full_report')
    p_doe.add_argument('--no_save', action='store_false', dest='salvar_graficos')
    p_doe.add_argument('-i', '--inputs', nargs='+', default=None)
    p_doe.add_argument('-o', '--outputs', nargs='+', default=None)
    p_doe.add_argument('--subset', nargs='+', default=None)
    p_doe.add_argument('--chunk_x', type=int, default=3)
    p_doe.add_argument('--chunk_y', type=int, default=2)

    # OAT
    p_oat = subparsers.add_parser('oat')
    p_oat.add_argument('-s', '--steps', type=int, default=11)
    p_oat.add_argument('-r', '--range', type=float, nargs=2, default=[0.8, 1.2])
    p_oat.add_argument('-i', '--inputs', nargs='+', default=None)
    p_oat.add_argument('-o', '--outputs', nargs='+', default=None)

    args = parser.parse_args()

    print(f"Carregando baseline: {args.baseline}...")
    try:
        aeronave_base = dt.standard_airplane(args.baseline)
    except Exception as e:
        print(f"Erro ao carregar aeronave: {e}")
        return

    vars_out = []
    if args.outputs:
        for v in args.outputs:
            if v in MASTER_OUTPUTS: vars_out.append(v)
    else:
        vars_out = MASTER_OUTPUTS
    
    if not vars_out:
        print("Erro: Nenhuma saída válida.")
        return

    if args.analysis_type == 'doe':
        vars_in_bounds = {}
        src = args.inputs if args.inputs else MASTER_DOE_BOUNDS
        for v in (src if args.inputs else MASTER_DOE_BOUNDS.keys()):
            if v in MASTER_DOE_BOUNDS: vars_in_bounds[v] = MASTER_DOE_BOUNDS[v]
        
        subset = args.subset if args.subset else MASTER_PAIRGRID_SUBSET
        
        executar_doe_analysis(aeronave_base, vars_in_bounds, vars_out, args.n_samples, 
                              plot_style=args.plot_style, plot_vars_subset=subset, output_dir=args.output_dir,
                              salvar_graficos=args.salvar_graficos, matrix_chunk_X=args.chunk_x, matrix_chunk_Y=args.chunk_y)

    elif args.analysis_type == 'oat':
        vars_in = []
        src = args.inputs if args.inputs else MASTER_OAT_INPUTS
        for v in src:
            if v in MASTER_OAT_INPUTS: vars_in.append(v)
            
        analise_parametrica(aeronave_base, vars_in, vars_out, tuple(args.range), args.steps, args.output_dir)

    print(f"Fim. Resultados em: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()