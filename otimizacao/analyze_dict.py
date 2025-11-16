import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.designTool as dt
import otimizacao.aerodinamica as aero
import otimizacao.pesos as pesos
import otimizacao.desempenho as desemp
import otimizacao.estabilidade as estab
import numpy as np

def print_dict(airplane):
    all_keys = sorted(set(airplane))
    for key in all_keys:
        val = airplane.get(key, '—')
        print(f"{key}: {val}")

with open("airplane_opt.json", "r") as file:
    airplane_opt = json.load(file)

dt.plot3d(airplane_opt)

airplane_opt['b'] = np.sqrt(airplane_opt['AR_w']*airplane_opt['S_w'])

# airplane_opt = aero.analise_aerodinamica(airplane_opt, show_results=False)

# print_dict(airplane_opt)

# pesos.analise_pesos(airplane_opt)

desemp.analise_desempenho(airplane_opt)

# estab.analise_estabilidade(airplane_opt)
