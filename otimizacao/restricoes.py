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

print(f"airplane_opt['W0'] = {airplane_opt['W0']} \t")
print(f"airplane_opt['deltaS_wlan'] = {airplane_opt['deltaS_wlan']} \t (>= 0)")
print(f"airplane_opt['SM_fwd'] = {airplane_opt['SM_fwd']} \t (<= 0.4)")
print(f"airplane_opt['SM_aft'] = {airplane_opt['SM_aft']} \t (>= 0.05)")
print(f"airplane_opt['CLv'] = {airplane_opt['CLv']} \t (<= 0.75)")
print(f"airplane_opt['frac_nlg_fwd'] = {airplane_opt['frac_nlg_fwd']} \t (<= 0.15)")
print(f"airplane_opt['frac_nlg_aft'] = {airplane_opt['frac_nlg_aft']} \t (>= 0.04)")
print(f"airplane_opt['alpha_tipback'] = {airplane_opt['alpha_tipback']*180/np.pi} \t (>= 15)")
print(f"airplane_opt['alpha_tailstrike'] = {airplane_opt['alpha_tailstrike']*180/np.pi} \t (>= 10)")
print(f"airplane_opt['phi_overturn'] = {airplane_opt['phi_overturn']*180/np.pi} \t (<= 63)")
print(f"airplane_opt['tank_excess'] = {airplane_opt['tank_excess']} \t (>= 0)")

# print(f"(airplane_opt['c_tank_c_w'] + airplane_opt['c_flap_c_wing']) = {(airplane_opt['c_tank_c_w'] + airplane_opt['c_flap_c_wing'])} \t (<= {(1 - airplane_opt['x_tank_c_w'])})")