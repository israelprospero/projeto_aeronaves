"""
PAINEL DE CONTROLE PARA O MDO ANALYSIS SUITE
---------------------------------------------

Este script importa as funções do 'mdo_analysis_suite.py' e permite
configurar e executar as análises editando variáveis Python, 
em vez de usar a linha de comando.

COMO USAR:
1. Altere as variáveis na seção "🎮 PAINEL DE CONTROLE".
2. Rode este script:
   $ python run_analysis.py
"""

# === Importações da Suite e Ferramentas ===
import mdo_analysis_suite as suite
import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import modules.designTool as dt

# ================================================================
# === 🎮 PAINEL DE CONTROLE ===
# (Edite esta seção para configurar sua análise)
# ================================================================

# 1. ESCOLHA A ANÁLISE:
#    Opções: 'doe' ou 'oat'
TIPO_ANALISE = 'doe'

# 2. DEFINA A AERONAVE DE REFERÊNCIA (BASELINE)
#    (Carregue o avião aqui)
#NOME_AERONAVE_REFERENCIA = "Fokker 100 (Padrão)"
#AERONAVE_REFERENCIA = dt.standard_airplane('fokker100')

NOME_AERONAVE_REFERENCIA = "Avião Configuração 1"
AERONAVE_REFERENCIA = dt.standard_airplane('my_airplane_1')

# 3. DEFINA A PASTA DE SAÍDA:
#    (Será criada se não existir)
PASTA_SAIDA = "meu_teste_doe_customizado"

# 4. CONFIGURAÇÕES COMUNS
#    (Deixe 'None' para usar a lista MESTRA completa do mdo_analysis_suite)
LISTA_SAIDAS = None
# LISTA_SAIDAS = None # Exemplo: Usar TODAS as saídas

# 5. CONFIGURAÇÕES ESPECÍFICAS DO 'DOE':
#    (Só é usado se TIPO_ANALISE == 'doe')
config_doe = {
    'n_samples': 50,
    'plot_style': 'full_report', # 'none', 'pairgrid', 'individual', etc.
    
    # Deixe 'None' para usar a lista MESTRA
    'inputs': None, 
    # 'inputs': None, # Exemplo: Usar TODOS os inputs
    
    # Deixe 'None' para usar o SUBSET MESTRE
    'subset': None,
    # 'subset': None, # Exemplo: Usar o subset padrão
    
    'chunk_x': 3,
    'chunk_y': 2,
    'salvar_graficos': True
}

# 6. CONFIGURAÇÕES ESPECÍFICAS DO 'OAT' (Paramétrica):
#    (Só é usado se TIPO_ANALISE == 'oat')
config_oat = {
    'steps': 11,
    'range': (0.8, 1.2), # Corrigido: 0.8 em vez de 0.
    
    # Deixe 'None' para usar a lista MESTRA
    'inputs': None,
    # 'inputs': None, # Exemplo: Usar TODOS os inputs
}
# ================================================================
# === FIM DO PAINEL DE CONTROLE === 
# (Não é necessário editar abaixo desta linha)
# ================================================================

# --- Funções Auxiliares de Validação ---

def _validar_lista(lista_pedida, lista_mestra, nome_var):
    """Valida uma lista de variáveis contra a lista mestra."""
    if lista_pedida is None:
        print(f"Usando lista MESTRA para '{nome_var}'.")
        return lista_mestra
    
    lista_validada = []
    print(f"Usando lista PERSONALIZADA para '{nome_var}': {lista_pedida}")
    for var in lista_pedida:
        if var in lista_mestra:
            lista_validada.append(var)
        else:
            print(f"   -> AVISO: Variável '{var}' não encontrada na lista mestra. Ignorando.")
    
    if not lista_validada:
        print(f"ERRO: Nenhuma variável válida encontrada para '{nome_var}'.")
        return None
    return lista_validada

def _validar_bounds_doe(lista_chaves_pedida, dict_mestre_bounds):
    """Valida e filtra o dicionário de limites do DOE."""
    if lista_chaves_pedida is None:
        print("Usando lista MESTRA para 'Entradas DOE'.")
        return dict_mestre_bounds
        
    bounds_validado = {}
    print(f"Usando lista PERSONALIZADA para 'Entradas DOE': {lista_chaves_pedida}")
    for var in lista_chaves_pedida:
        if var in dict_mestre_bounds:
            bounds_validado[var] = dict_mestre_bounds[var]
        else:
            print(f"   -> AVISO: Variável '{var}' não encontrada nos BOUNDS mestres. Ignorando.")
            
    if not bounds_validado:
        print("ERRO: Nenhuma variável de entrada válida encontrada para 'Entradas DOE'.")
        return None
    return bounds_validado


# --- Bloco Principal de Execução ---

if __name__ == "__main__":
    
    print("--- Iniciando execução via 'run_analysis.py' ---")
    
    # 1. Carregar Aeronave Base
    print(f"Carregando aeronave de referência: {NOME_AERONAVE_REFERENCIA}...")
    aeronave_base = AERONAVE_REFERENCIA
    if aeronave_base is None:
        exit("ERRO FATAL: A aeronave de referência (AERONAVE_REFERENCIA) não foi carregada.")

    # 2. Criar pasta de saída, se necessário
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path_completo = os.path.join(BASE_DIR, PASTA_SAIDA)
    
    if not os.path.exists(output_path_completo):
        os.makedirs(output_path_completo)
        print(f"Criado diretório de saída: '{output_path_completo}'")
    else:
        print(f"Usando diretório de saída existente: '{output_path_completo}'")


    # 3. Validar Saídas (Comum a ambos)
    saidas_finais = _validar_lista(LISTA_SAIDAS, suite.MASTER_OUTPUTS, "Saídas")
    if saidas_finais is None:
        exit("Execução abortada: Nenhuma variável de saída válida.")
        
    # 4. Decidir qual análise rodar
    
    if TIPO_ANALISE == 'doe':
        print("\n--- Configurando Análise DOE ---")
        
        # Validar Entradas DOE
        bounds_finais = _validar_bounds_doe(config_doe['inputs'], suite.MASTER_DOE_BOUNDS)
        if bounds_finais is None:
            exit("Execução abortada: Nenhuma variável de entrada válida para o DOE.")
            
        # Validar Subset do Pairgrid
        subset_final = config_doe['subset']
        if subset_final is None:
            print("Usando SUBSET padrão para o PairGrid de resumo.")
            subset_final = suite.MASTER_PAIRGRID_SUBSET
        else:
            print(f"Usando SUBSET PERSONALIZADO para o PairGrid: {subset_final}")
            
        # Chamar a função do MÓDULO
        suite.executar_doe_analysis(
            baseline_airplane=aeronave_base,
            variaveis_entrada_bounds=bounds_finais,
            variaveis_saida=saidas_finais,
            n_samples=config_doe['n_samples'],
            plot_style=config_doe['plot_style'],
            plot_vars_subset=subset_final,
            output_dir=output_path_completo,
            salvar_graficos=config_doe['salvar_graficos'],
            matrix_chunk_X=config_doe['chunk_x'],
            matrix_chunk_Y=config_doe['chunk_y']
        )
        
    elif TIPO_ANALISE == 'oat':
        print("\n--- Configurando Análise OAT (Paramétrica) ---")
        
        # Validar Entradas OAT
        entradas_finais_oat = _validar_lista(config_oat['inputs'], suite.MASTER_OAT_INPUTS, "Entradas OAT")
        if entradas_finais_oat is None:
            exit("Execução abortada: Nenhuma variável de entrada válida para o OAT.")
        
        # Chamar a função do MÓDULO
        suite.analise_parametrica(
            baseline_airplane=aeronave_base,
            variaveis_entrada=entradas_finais_oat,
            variaveis_saida=saidas_finais,
            intervalo_percentual=tuple(config_oat['range']),
            num_passos=config_oat['steps'],
            output_dir=output_path_completo
        )
        
    else:
        print(f"ERRO: TIPO_ANALISE '{TIPO_ANALISE}' não reconhecido.")
        print("Por favor, escolha 'doe' ou 'oat' no PAINEL DE CONTROLE.")

    print(f"\n--- Execução do 'run_analysis.py' finalizada ---")
    print(f"Resultados salvos no diretório: '{os.path.abspath(output_path_completo)}'")