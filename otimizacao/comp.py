import json
import sys
import os
import copy  # <--- Adicionado para criar cópia independente
import numpy as np

# ... (Seus imports de módulos permanecem aqui) ...
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.designTool as dt
import otimizacao.aerodinamica as aero

# --- CONFIGURAÇÃO INICIAL ---
json_path = os.path.join(parent_dir, "airplane_opt.json")
print(f"Lendo arquivo de: {json_path}")

with open(json_path, "r") as file:
    data_raw = json.load(file)

# --- CENÁRIO 1: BASELINE (Original) ---
airplane_base = copy.deepcopy(data_raw)

# Cálculos padrão que devem ocorrer no original também
airplane_base['b'] = np.sqrt(airplane_base['AR_w']*airplane_base['S_w'])

# Roda as análises no original
dt.geometry(airplane_base)
dt.analyze(airplane_base)

# --- CENÁRIO 2: MODIFICADO ---
airplane_mod = copy.deepcopy(airplane_base) # Copia do base já calculado para garantir consistência

# >>> SUAS ALTERAÇÕES AQUI <<<
airplane_mod['Lb_v'] = 0.43
airplane_mod['sweep_v'] = 0.523598775598333

# Recalcula geometria e análise com os novos parâmetros
dt.geometry(airplane_mod)
dt.analyze(airplane_mod)

# --- COMPARAÇÃO DE RESULTADOS ---

print("\n" + "="*95)
print(f"{'PARÂMETRO':<25} | {'ORIGINAL':<15} | {'NOVO':<15} | {'DELTA':<10} | {'% MUDANÇA':<10}")
print("="*95)

all_keys = set(airplane_base.keys()).union(set(airplane_mod.keys()))

for key in sorted(all_keys):
    val_base = airplane_base.get(key)
    val_mod = airplane_mod.get(key)

    if val_base == val_mod:
        continue

    # Verificação estrita: É número escalar? (Ignora listas, arrays e dicts para evitar erro de visualização)
    is_number = (
        isinstance(val_base, (int, float, np.number)) and 
        isinstance(val_mod, (int, float, np.number)) and
        not isinstance(val_base, (list, dict, np.ndarray)) # Garante que não é vetor
    )

    if is_number:
        delta = val_mod - val_base
        
        if val_base != 0:
            pct = (delta / val_base) * 100
            pct_str = f"{pct:+.2f}%"
        else:
            pct_str = "N/A"
            
        # Só mostra se a diferença for relevante (ex: > 0.000001)
        if abs(delta) > 1e-6:
            print(f"{key:<25} | {val_base:<15.4f} | {val_mod:<15.4f} | {delta:<+10.4f} | {pct_str:<10}")
    
    else:
        # Tratamento para TEXTO/LISTAS: Corta o texto se for muito grande para não quebrar a tabela
        str_base = str(val_base)
        str_mod = str(val_mod)
        
        # Função lambda rápida para cortar strings maiores que 12 caracteres
        truncate = lambda s: (s[:12] + '..') if len(s) > 12 else s
        
        print(f"{key:<25} | {truncate(str_base):<15} | {truncate(str_mod):<15} | {'---':<10} | {'---':<10}")

print("="*95 + "\n")