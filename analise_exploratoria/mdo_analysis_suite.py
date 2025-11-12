"""
MDO Analysis Suite
==================

Conjunto de ferramentas para análise de aeronaves integrado com o designTool.py.

...
"""

# === IMPORTAÇÕES ===
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
import copy
import warnings
import os
import sys
import argparse
from tabulate import tabulate
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import modules.designTool as dt

# ================================================================
# === DEFINIÇÕES MESTRAS (MASTER DEFINITIONS) ===
# ================================================================

# TODO: definir saidas
# TODO: definir bounds
# TODO: definir bounds para perfil da asa
# TODO: definir bounds para perfil da empenagem
# TODO: restricao: landing gear > diametro da nacelle
# TODO: restricao: EH nao deve ficar para fora do aviao
# TODO: restricao: EV nao deve ficar para fora do aviao
# TODO: restricao: flap/tanque/aileron nao devem ocupar 100% da asa

def get_perc_var(val):
    perc = 0.1
    if val > 0:
        return val - perc * val, val + perc * val
    else:
        return val + perc * val, val - perc * val

airplane_ref = dt.standard_airplane('my_airplane_1')
MASTER_DOE_BOUNDS = {
    'S_w':              [70, 105],
    'AR_w':             [7, 11],
    'taper_w':          [0.2, 0.4],
    'sweep_w':          [20*np.pi/180, 30*np.pi/180],
    'dihedral_w':       [2*np.pi/180, 6.5*np.pi/180],
    'xr_w':             list(get_perc_var(airplane_ref['xr_w'])),
    'zr_w':             list(get_perc_var(airplane_ref['zr_w'])),
    'tcr_w':            list(get_perc_var(airplane_ref['tcr_w'])),
    'tct_w':            list(get_perc_var(airplane_ref['tct_w'])),
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
    'zr_v':             list(get_perc_var(airplane_ref['zr_v'])),
    'x_n':              [airplane_ref['xr_w'] - 3, airplane_ref['xr_w'] + 4],
    'y_n':              list(get_perc_var(airplane_ref['y_n'])),
    'z_n':              list(get_perc_var(airplane_ref['z_n'])),
    'L_n':              [3, 4.5],
    'D_n':              [1.5, 2.3],
    'x_nlg':            [2.5, 5.5],
    'x_mlg':            [airplane_ref['xr_w'] + 1.7, airplane_ref['xr_w'] + 3.9],
    'y_mlg':            [2, 4],
    'z_lg':             [-4, -2],
    'x_tailstrike':     list(get_perc_var(airplane_ref['x_tailstrike'])),
    'z_tailstrike':     list(get_perc_var(airplane_ref['z_tailstrike'])),
    'c_tank_c_w':       list(get_perc_var(airplane_ref['c_tank_c_w'])),
    'b_tank_b_w_end':   list(get_perc_var(airplane_ref['b_tank_b_w_end'])),
    'c_flap_c_wing':    list(get_perc_var(airplane_ref['c_flap_c_wing'])),
    'b_flap_b_wing':    list(get_perc_var(airplane_ref['b_flap_b_wing'])),
    'c_ail_c_wing':     list(get_perc_var(airplane_ref['c_ail_c_wing'])),
    'b_ail_b_wing':     list(get_perc_var(airplane_ref['b_ail_b_wing']))
}

MASTER_OAT_INPUTS = [
    'S_w',
    'AR_w',
    'taper_w',
    'sweep_w',
    'dihedral_w',
    'xr_w',
    'zr_w',
    'tcr_w',
    'tct_w',
    'Cht',
    'Lc_h',
    'AR_h',
    'taper_h',
    'sweep_h',
    'dihedral_h',
    'zr_h',
    'tcr_h',
    'tct_h',
    'Cvt',
    'Lb_v',
    'AR_v',
    'taper_v',
    'sweep_v',
    'zr_v',
    'x_n',
    'y_n',
    'z_n',
    'L_n',
    'D_n',
    'x_nlg',
    'x_mlg',
    'y_mlg',
    'z_lg',
    'x_tailstrike',
    'z_tailstrike',
    'c_tank_c_w',
    'b_tank_b_w_end',
    'c_flap_c_wing',
    'b_flap_b_wing',
    'c_ail_c_wing',
    'b_ail_b_wing'
]

MASTER_OUTPUTS = [
    'W0', 'W_empty', 'W_fuel', 'T0', 'DOC', 'deltaS_wlan', 'tank_excess',
    'CLv', 'SM_fwd', 'SM_aft', 'alpha_tipback', 'alpha_tailstrike', 'phi_overturn'
]

MASTER_PAIRGRID_SUBSET = [
    'W0', 'W_empty', 'W_fuel', 'T0', 'DOC', 'deltaS_wlan', 'tank_excess',
    'CLv', 'SM_fwd', 'SM_aft', 'alpha_tipback', 'alpha_tailstrike', 'phi_overturn'
]
# ================================================================

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
        print(f"AVISO: 'num_passos' deve ser ímpar e >= 3. "
              f"Ajustando para {max(3, num_passos + (1 - num_passos % 2))}.")
        num_passos = max(3, num_passos + (1 - num_passos % 2))

    print("--- Iniciando Análise Paramétrica (OAT) Robusta ---")
    
    # Criar diretório de saída se não existir
    output_plots_dir = os.path.join(output_dir, "oat_plots_individuais")
    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)
        print(f"Criado diretório: {output_plots_dir}")

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
        print(f"ERRO FATAL: A análise da aeronave de baseline falhou. "
              f"Não é possível continuar. Erro: {e}")
        return None, None
    
    resultados_grafico = {}
    dados_para_csv = []
    df = None 

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
                print(f"ERRO: Análise falhou para {var_in} = {valor_perturbado}. Erro: {e}. Adicionando 'None'.")
                for var_out in variaveis_saida:
                    dados_saida_var[var_out].append(None)
                    linha_csv[var_out] = None
            
            dados_para_csv.append(linha_csv)

        resultados_grafico[var_in] = {
            'vetor_normalizado_x': vetor_normalizado, 'vetor_absoluto_x': vetor_absoluto, 'dados_saida': dados_saida_var
        }

    print("--- Análise concluída. Gerando gráficos. ---")

    for var_out in variaveis_saida:
        plt.figure(figsize=(10, 6))
        for var_in in variaveis_entrada:
            if var_in in resultados_grafico:
                eixo_x = resultados_grafico[var_in]['vetor_normalizado_x']
                eixo_y = resultados_grafico[var_in]['dados_saida'][var_out]
                plt.plot(eixo_x, eixo_y, 'o-', label=f"{var_in}")
        
        plt.title(f"Análise de Sensibilidade (OAT) para: {var_out}", fontsize=16)
        plt.xlabel("Variação % em relação ao valor de referência", fontsize=12)
        plt.ylabel(f"Valor de Saída: {var_out}", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axvline(x=100, color='red', linestyle=':', linewidth=2, label='Baseline (100%)')
        
        plot_filename = os.path.join(output_plots_dir, f"oat_plot__{var_out}.png")
        try:
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Gráfico salvo em: {plot_filename}")
        except Exception as e:
            print(f"ERRO ao salvar gráfico {plot_filename}: {e}")
        plt.close()

    print(f"--- Gráficos salvos em '{output_plots_dir}'. ---")
    
    try:
        print(f"\nSalvando dados detalhados em '{csv_filename}'...")
        df = pd.DataFrame(dados_para_csv)
        colunas_ordem = ['variavel_entrada', 'percentual_entrada', 'valor_entrada'] + variaveis_saida
        df = df[colunas_ordem]
        df.to_csv(csv_filename, index=False, float_format='%.5e')
        print(f"Arquivo '{csv_filename}' salvo com sucesso.")
    except Exception as e:
        print(f"ERRO ao salvar CSV: {e}")

    print("\n--- Tabela de Análise de Sensibilidade (no Ponto de Referência) ---")
    
    tabela_sensibilidade = []
    headers = ["Variável Saída (y)", "Variável Entrada (x)", "Sensib. Absoluta (dy/dx)", "Sensib. Relativa (dy/y*)/(dx/x*)"]
    
    idx_base = num_passos // 2
    idx_pre = idx_base - 1
    idx_post = idx_base + 1
    
    for var_out in variaveis_saida:
        y_star = y_baseline_valores.get(var_out)
        if y_star is None or y_star == 0:
            print(f"AVISO: Pulando sensibilidade para '{var_out}' (valor baseline nulo ou zero).")
            continue
            
        for var_in in variaveis_entrada:
            if var_in not in resultados_grafico: continue
            x_star = baseline_airplane[var_in]
            res = resultados_grafico[var_in]
            
            try:
                x_pre, x_post = res['vetor_absoluto_x'][idx_pre], res['vetor_absoluto_x'][idx_post]
                y_pre, y_post = res['dados_saida'][var_out][idx_pre], res['dados_saida'][var_out][idx_post]
                
                if y_pre is None or y_post is None: raise TypeError("Ponto de análise falhou.")
                delta_y, delta_x = y_post - y_pre, x_post - x_pre
                if delta_x == 0: raise ZeroDivisionError("delta_x é zero.")

                abs_sens = delta_y / delta_x
                rel_sens = abs_sens * (x_star / y_star)
                tabela_sensibilidade.append([var_out, var_in, abs_sens, rel_sens])
            except Exception as e:
                tabela_sensibilidade.append([var_out, var_in, f"ERRO ({e})", f"ERRO ({e})"])
    
    print(tabulate(tabela_sensibilidade, headers=headers, floatfmt=".4f", tablefmt="grid"))
    print("--- Fim da Análise Paramétrica ---")

    return resultados_grafico, df

# =============================================================================
# === FUNÇÃO 2: DESIGN OF EXPERIMENTS (DOE) ===
# =============================================================================

# --- Funções Auxiliares de Plotagem ---

def corrdot(*args, **kwargs):
    """(Esta função é IDÊNTICA à versão anterior)"""
    x_data, y_data = args[0], args[1]
    valid_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    
    if len(valid_data) < 2: corr_r = np.nan
    else: corr_r = valid_data['x'].corr(valid_data['y'], 'pearson')
        
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca(); ax.set_axis_off()
    
    if pd.isna(corr_r):
        ax.annotate("NaN", [.5, .5,], xycoords="axes fraction", ha='center', va='center', fontsize=20, color='gray')
        return

    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm", vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,], xycoords="axes fraction", ha='center', va='center', fontsize=font_size)

def _gerar_pairgrid(df_plot, title, filename, salvar_graficos, show_plot=True):
    """(Esta função é IDÊNTICA à versão anterior)"""
    try:
        n_plot_vars = len(df_plot.columns)
        altura_subplot = max(1.5, min(3.0, 15 / n_plot_vars))
        
        sns.set(style='white', font_scale=1.1)
        fig = sns.PairGrid(df_plot, diag_sharey=False, height=altura_subplot)
        fig.map_diag(sns.histplot, kde=True)
        fig.map_lower(sns.regplot, lowess=True, scatter_kws={'alpha': 0.3}, line_kws={'color': 'black', 'lw': 2})
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
        print(f"   ERRO ao gerar PairGrid para '{title}': {e}")
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
    """
    (Esta função é IDÊNTICA à versão anterior)
    """
    
    print(f"--- Iniciando DOE com {n_samples} amostras ---")
    
    # --- Configuração de Caminhos de Saída ---
    csv_filename = os.path.join(output_dir, f"doe_results_{plot_style}_{n_samples}s.csv")
    individual_plots_dir = os.path.join(output_dir, "doe_plots_individuais")
    matrix_plots_dir = os.path.join(output_dir, "doe_plots_matrizes")
    pairgrid_filename = os.path.join(output_dir, "doe_correlation_plot.png")

    if salvar_graficos and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Criado diretório de saída: {output_dir}")

    # 1. Preparar o problema para o Pymoo
    variaveis_entrada_nomes = list(variaveis_entrada_bounds.keys())
    
    if not variaveis_entrada_nomes:
        print("ERRO FATAL: Nenhuma variável de entrada válida foi selecionada para o DOE.")
        return None
    if not variaveis_saida:
        print("ERRO FATAL: Nenhuma variável de saída válida foi selecionada.")
        return None
        
    n_var = len(variaveis_entrada_nomes)
    lb = [variaveis_entrada_bounds[key][0] for key in variaveis_entrada_nomes]
    ub = [variaveis_entrada_bounds[key][1] for key in variaveis_entrada_nomes]

    problem = Problem(n_var=n_var, xl=lb, xu=ub)

    # 2. Gerar as amostras
    print(f"Gerando amostras com {sampler_method.__class__.__name__}...")
    X_samples = sampler_method(problem, n_samples).get("X")

    # 3. Iterar pelas amostras
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
        except Exception as e:
            output_dict = {var_out: np.nan for var_out in variaveis_saida}
        
        all_results_data.append({**input_dict, **output_dict})

    print(f"--- Análise DOE concluída ({n_samples} amostras executadas) ---")

    # 4. Criar DataFrame e salvar em CSV
    df = pd.DataFrame(all_results_data)
    
    if csv_filename and salvar_graficos:
        try:
            df.to_csv(csv_filename, index=False, float_format='%.5e')
            print(f"Resultados completos salvos em '{csv_filename}'")
        except Exception as e:
            print(f"ERRO ao salvar CSV: {e}")

    # 5. Gerar os gráficos
    if not salvar_graficos:
        print("Salvamento de gráficos e CSV desativado.")
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # --- PARTE 1: GRÁFICOS INDIVIDUAIS ---
        if plot_style == 'individual' or plot_style == 'full_report':
            print("Gerando gráficos individuais (Entrada vs. Saída)...")
            if not os.path.exists(individual_plots_dir): os.makedirs(individual_plots_dir)
            n_plots = len(variaveis_entrada_nomes) * len(variaveis_saida)
            print(f"Total de {n_plots} gráficos a serem salvos em '{individual_plots_dir}/'")
            
            plot_count = 1
            for var_out in variaveis_saida:
                for var_in in variaveis_entrada_nomes:
                    if df[var_in].isnull().all() or df[var_out].isnull().all():
                        plot_count += 1
                        continue
                    
                    plt.figure(figsize=(8, 5))
                    sns.regplot(data=df, x=var_in, y=var_out, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2})
                    plt.title(f"{var_out} vs. {var_in}", fontsize=16)
                    plt.xlabel(f"Entrada: {var_in}", fontsize=12)
                    plt.ylabel(f"Saída: {var_out}", fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    safe_filename = os.path.join(individual_plots_dir, f"plot__{var_out}_vs_{var_in}.png")
                    plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    plot_count += 1
            print(f"Gráficos individuais salvos com sucesso.")

        # --- PARTE 2: GRÁFICOS (PairGrid) ---
        if plot_style == 'pairgrid' or plot_style == 'full_report':
            print("Gerando gráfico de matriz de correlação (PairGrid)...")
            
            if plot_vars_subset:
                plot_vars_existentes = [v for v in plot_vars_subset if v in df.columns]
                if not plot_vars_existentes:
                     print("AVISO: Nenhuma das variáveis do '--subset' foi encontrada. Pulando PairGrid.")
                     df_plot = pd.DataFrame() 
                else:
                    print(f"Usando subset para PairGrid: {plot_vars_existentes}")
                    df_plot = df[plot_vars_existentes]
            else:
                print("AVISO: '--subset' não definido. Plotando PairGrid com TODAS as variáveis.")
                df_plot = df

            if df_plot.empty:
                print("Nenhuma variável de plotagem selecionada. Pulando PairGrid.")
            else:
                _gerar_pairgrid(df_plot, "Matriz de Correlação DOE", pairgrid_filename, 
                                salvar_graficos, show_plot=(plot_style == 'pairgrid'))

        # --- PARTE 3: MATRIZES COMBINADAS ---
        if plot_style == 'matrix_combinations' or plot_style == 'full_report':
            print("Gerando gráficos de matrizes combinadas (X vs Y)...")
            if not os.path.exists(matrix_plots_dir): os.makedirs(matrix_plots_dir)

            input_chunks = [variaveis_entrada_nomes[i:i + matrix_chunk_X] for i in range(0, len(variaveis_entrada_nomes), matrix_chunk_X)]
            output_chunks = [variaveis_saida[i:i + matrix_chunk_Y] for i in range(0, len(variaveis_saida), matrix_chunk_Y)]
            
            total_plots = len(input_chunks) * len(output_chunks)
            print(f"Serão geradas {total_plots} matrizes (blocos de {matrix_chunk_X}x{matrix_chunk_Y})...")
            
            plot_count = 1
            for i, in_chunk in enumerate(input_chunks):
                for j, out_chunk in enumerate(output_chunks):
                    plot_vars_subset = in_chunk + out_chunk
                    df_plot = df[plot_vars_subset]
                    title = f"Matriz Combinada (Entradas Bloco {i+1} / Saídas Bloco {j+1})"
                    filename = os.path.join(matrix_plots_dir, f"matriz_X{i+1}_Y{j+1}.png")
                    
                    print(f"   Gerando matriz {plot_count}/{total_plots}...")
                    
                    _gerar_pairgrid(df_plot, title, filename, salvar_graficos, show_plot=False)
                    plot_count += 1
            print(f"Gráficos de matrizes combinadas salvos em '{matrix_plots_dir}'")
            
        # --- PARTE 4: OUTROS ESTILOS ---
        elif plot_style == 'none':
            print("Geração de gráficos pulada por solicitação.")
        elif plot_style not in ['individual', 'full_report', 'pairgrid', 'matrix_combinations']:
            print(f"AVISO: '{plot_style}' não é um 'plot_style' reconhecido.")

    return df

# =============================================================================
# === BLOCO PRINCIPAL DE EXECUÇÃO (Interface de Linha de Comando) ===
# =============================================================================

def main():
    """
    Função principal que gerencia a interface de linha de comando (CLI)
    para selecionar e executar as análises.
    """
    
    # --- Configuração do Parser Principal ---
    parser = argparse.ArgumentParser(
        description="MDO Analysis Suite - Executa análises OAT ou DOE.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Argumentos Globais ---
    parser.add_argument(
        '-dir', '--output_dir',
        type=str,
        default="mdo_analysis_output",
        help="Diretório de saída principal para salvar CSVs e gráficos.\n(Padrão: 'mdo_analysis_output')"
    )
    
    parser.add_argument(
        '-b', '--baseline',
        type=str,
        default='fokker100',
        help="Nome da aeronave de referência para carregar via dt.standard_airplane().\n(Padrão: 'fokker100')"
    )

    subparsers = parser.subparsers(dest='analysis_type', required=True, help="O tipo de análise a ser executada.")

    # --- Sub-parser para 'doe' ---
    parser_doe = subparsers.add_parser('doe', help="Executa uma análise de Design of Experiments (DOE).", description="Executa uma análise DOE (ex: LHS) com N amostras.")
    parser_doe.add_argument('-n', '--n_samples', type=int, default=100, help="Número de amostras para o DOE. (Padrão: 100)")
    parser_doe.add_argument('-p', '--plot_style', type=str, default='full_report', choices=['none', 'pairgrid', 'individual', 'matrix_combinations', 'full_report'], help="O tipo de relatório gráfico. (Padrão: 'full_report')")
    parser_doe.add_argument('--no_save', action='store_false', dest='salvar_graficos', help="Desativa o salvamento de gráficos e CSVs.")
    parser_doe.add_argument('-i', '--inputs', nargs='+', default=None, help="Lista de variáveis de ENTRADA para o DOE (ex: S_w AR_w). \n(Padrão: usa TODAS as variáveis da lista mestra)")
    parser_doe.add_argument('-o', '--outputs', nargs='+', default=None, help="Lista de variáveis de SAÍDA para monitorar (ex: W0 DOC). \n(Padrão: usa TODAS as variáveis da lista mestra)")
    parser_doe.add_argument('--subset', nargs='+', default=None, help="Lista de variáveis (entradas+saídas) para o PairGrid de resumo. \n(Padrão: usa o subset default da lista mestra)")
    parser_doe.add_argument('--chunk_x', type=int, default=3, help="Número de entradas (X) por gráfico de matriz. (Padrão: 3)")
    parser_doe.add_argument('--chunk_y', type=int, default=2, help="Número de saídas (Y) por gráfico de matriz. (Padrão: 2)")

    # --- Sub-parser para 'oat' ---
    parser_oat = subparsers.add_parser('oat', help="Executa uma análise paramétrica One-at-a-Time (OAT).", description="Executa uma varredura OAT em torno do baseline.")
    parser_oat.add_argument('-s', '--steps', type=int, default=11, help="Número de passos da varredura (ÍMPAR). (Padrão: 11)")
    parser_oat.add_argument('-r', '--range', type=float, nargs=2, default=[0.8, 1.2], metavar=('MIN', 'MAX'), help="Intervalo percentual da varredura. (Padrão: 0.8 1.2)")
    parser_oat.add_argument('-i', '--inputs', nargs='+', default=None, help="Lista de variáveis de ENTRADA para o OAT (ex: S_w AR_w). \n(Padrão: usa TODAS as variáveis da lista mestra)")
    parser_oat.add_argument('-o', '--outputs', nargs='+', default=None, help="Lista de variáveis de SAÍDA para monitorar (ex: W0 DOC). \n(Padrão: usa TODAS as variáveis da lista mestra)")

    # --- Analisa os argumentos da linha de comando ---
    args = parser.parse_args()

    # --- Carregar Aeronave Base ---
    print(f"Carregando aeronave de referência: {args.baseline}...")
    try:
        aeronave_base = dt.standard_airplane(args.baseline)
        print("Aeronave carregada com sucesso.")
    except Exception as e:
        print(f"ERRO FATAL: Não foi possível carregar a aeronave de referência '{args.baseline}'.")
        print(f"Verifique se o nome está correto e disponível em dt.standard_airplane().")
        print(f"Erro: {e}")
        return 

    # --- Lógica de Validação e Seleção de Saídas ---
    variaveis_saida_selecionadas = []
    if args.outputs is None:
        print("Usando lista MESTRA de variáveis de SAÍDA.")
        variaveis_saida_selecionadas = MASTER_OUTPUTS
    else:
        print(f"Usando lista PERSONALIZADA de variáveis de SAÍDA: {args.outputs}")
        for var in args.outputs:
            if var in MASTER_OUTPUTS:
                variaveis_saida_selecionadas.append(var)
            else:
                print(f"   -> AVISO: Variável de saída '{var}' não encontrada na lista mestra. Ignorando.")
        if not variaveis_saida_selecionadas:
            print("ERRO: Nenhuma variável de saída válida foi selecionada. Abortando.")
            return 

    # --- Execução da Análise Solicitada ---
    
    if args.analysis_type == 'doe':
        
        # --- Lógica de Validação e Seleção (DOE Inputs) ---
        variaveis_entrada_bounds_doe = {}
        input_list_source = args.inputs
        
        if input_list_source is None:
            print("Usando lista MESTRA de variáveis de ENTRADA (DOE).")
            variaveis_entrada_bounds_doe = MASTER_DOE_BOUNDS
        else:
            print(f"Usando lista PERSONALIZADA de variáveis de ENTRADA (DOE): {input_list_source}")
            for var in input_list_source:
                if var in MASTER_DOE_BOUNDS:
                    variaveis_entrada_bounds_doe[var] = MASTER_DOE_BOUNDS[var]
                else:
                    print(f"   -> AVISO: Variável de entrada '{var}' não encontrada na lista mestra de BOUNDS. Ignorando.")
        
        if not variaveis_entrada_bounds_doe:
            print("ERRO: Nenhuma variável de entrada válida foi selecionada para o DOE. Abortando.")
            return

        # --- Lógica de Seleção (DOE Pairgrid Subset) ---
        subset_selecionado = args.subset
        if subset_selecionado is None:
            print("Usando SUBSET padrão para o PairGrid de resumo.")
            subset_selecionado = MASTER_PAIRGRID_SUBSET
        else:
            print(f"Usando SUBSET PERSONALIZADO para o PairGrid: {subset_selecionado}")

        print(f"Iniciando análise 'doe' com {args.n_samples} amostras...")
        executar_doe_analysis(
            baseline_airplane=aeronave_base,
            variaveis_entrada_bounds=variaveis_entrada_bounds_doe,
            variaveis_saida=variaveis_saida_selecionadas,
            n_samples=args.n_samples,
            plot_style=args.plot_style,
            plot_vars_subset=subset_selecionado,
            output_dir=args.output_dir,
            salvar_graficos=args.salvar_graficos,
            matrix_chunk_X=args.chunk_x,
            matrix_chunk_Y=args.chunk_y
        )

    elif args.analysis_type == 'oat':
        
        # --- Lógica de Validação e Seleção (OAT Inputs) ---
        variaveis_entrada_oat = []
        input_list_source = args.inputs
        
        if input_list_source is None:
            print("Usando lista MESTRA de variáveis de ENTRADA (OAT).")
            variaveis_entrada_oat = MASTER_OAT_INPUTS
        else:
            print(f"Usando lista PERSONALIZADA de variáveis de ENTRADA (OAT): {input_list_source}")
            for var in input_list_source:
                if var in MASTER_OAT_INPUTS: 
                    variaveis_entrada_oat.append(var)
                else:
                    print(f"   -> AVISO: Variável de entrada '{var}' não encontrada na lista mestra de OAT. Ignorando.")
        
        if not variaveis_entrada_oat:
            print("ERRO: Nenhuma variável de entrada válida foi selecionada para o OAT. Abortando.")
            return

        print(f"Iniciando análise 'oat' com {args.steps} passos...")
        analise_parametrica(
            baseline_airplane=aeronave_base,
            variaveis_entrada=variaveis_entrada_oat,
            variaveis_saida=variaveis_saida_selecionadas,
            intervalo_percentual=tuple(args.range),
            num_passos=args.steps,
            output_dir=args.output_dir
        )

    print(f"\n--- Execução do script finalizada ---")
    print(f"Resultados salvos no diretório: '{os.path.abspath(args.output_dir)}'")


if __name__ == "__main__":
    main()