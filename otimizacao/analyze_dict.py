import json
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.designTool as dt
import otimizacao.aerodinamica as aero
import otimizacao.pesos as pesos
import otimizacao.desempenho as desemp
import otimizacao.estabilidade as estab
import numpy as np

'''#------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
output_path = "airplane_opt_updated.json"
with open(output_path, 'w') as f:
    json.dump(airplane_opt, f, indent=4, cls=NumpyEncoder)
print(f"Arquivo salvo com sucesso em: {output_path}")
#------------------------------------------------------------------------------'''

def print_dict(airplane):
    all_keys = sorted(set(airplane))
    for key in all_keys:
        val = airplane.get(key, '—')
        print(f"{key}: {val}")

json_path = os.path.join(parent_dir, "airplane_orig.json")

print(f"Lendo arquivo de: {json_path}")

with open(json_path, "r") as file:
    airplane_orig = json.load(file)

json_path = os.path.join(parent_dir, "airplane_opt_updated.json")

print(f"Lendo arquivo de: {json_path}")

with open(json_path, "r") as file:
    airplane_opt = json.load(file)

airplane_opt['b'] = np.sqrt(airplane_opt['AR_w']*airplane_opt['S_w'])

#airplane_opt['Lb_v'] = 0.43
#airplane_opt['sweep_v'] = 0.523598775598333

#dt.geometry(airplane_opt)
#dt.analyze(airplane_opt)
#print_dict(airplane_opt)

#dt.plot3d(airplane_opt)

airplane_opt = aero.analise_aerodinamica(airplane_opt, airplane_original=None,show_results=True) #airplane_original=airplane_orig

#pesos.analise_pesos(airplane_opt)

# desemp.analise_desempenho(airplane_opt)

# estab.analise_estabilidade(airplane_opt)
