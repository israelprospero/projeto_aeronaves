import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import designTool as dt
from modules import utils as m
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import print_fuel_table

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi

def analise_pesos(airplane):
    
    W0 = airplane['W0']
    W_empty = airplane['W_empty']
    W_fuel = airplane['W_fuel']
    
    print('Weights in kgf:')
    print('W0: %d '%(W0/dt.gravity))
    print('W_empty : %d '%(W_empty/dt.gravity))
    print('W_fuel : %d '%(W_fuel/dt.gravity))
    print('W_payload : %d '%(airplane['W_payload']/dt.gravity))
    print('W_crew : %d '%(airplane['W_crew']/dt.gravity))

    for key in ['W_w','W_h','W_v','W_f','W_nlg','W_mlg','W_eng','W_allelse']:
        print(key +': %d '%(airplane[key]/dt.gravity))

    # Pie Chart
    W_allelse = airplane['W_allelse']
    W_eng = airplane['W_eng']
    W_f = airplane['W_f']
    W_h = airplane['W_h']
    W_mlg = airplane['W_mlg']
    W_nlg = airplane['W_nlg']
    W_v = airplane['W_v']
    W_w = airplane['W_w']

    labels = [r'$W_{payload}$', r'$W_{crew}$', r'$W_{fuel}$', r'$W_{allelse}$', r'$W_{eng}$',
            r'$W_{f}$', r'$W_{h}$', r'$W_{mlg}$', r'$W_{nlg}$', r'$W_{v}$', r'$W_{w}$']
    valores = [airplane['W_payload'], airplane['W_crew'], W_fuel, W_allelse, W_eng, W_f, W_h, W_mlg, W_nlg, W_v, W_w]
    colors = ['#00008B', '#006400', '#DBB40C', '#DC143C', '#F97306', '#AAFF32',
            '#580F41', '#06C2AC', '#E50000', '#D2691E', '#FFFF14']

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        valores,
        colors=colors,
        autopct='%1.2f%%',
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 10},
        pctdistance=0.8
    )

    ax.legend(
        wedges,
        labels,
        title='Componentes',
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        title_fontsize=12
    )

    ax.axis('equal')
    plt.tight_layout()
    plt.savefig('weight_breakdown_airplane1.png', dpi=300)
    plt.show()

    #Table
    print_fuel_table(airplane)

    # L/D and TSFC
    _, _, CL_cruise, CD_cruise, C_cruise, L_D_max, C_loiter, CL_alt, CD_alt, C_altcruise = dt.fuel_weight(W0, airplane, airplane['range_cruise'], update_Mf_hist=False)
    L_D_cruise = CL_cruise/CD_cruise
    L_D_loiter = L_D_max
    L_D_alt = CL_alt/CD_alt

    print('CL_cruise = ', CL_cruise)
    print('CD_cruise =', CD_cruise)
    print('L_D_cruise =', L_D_cruise)
    print('C_cruise =', C_cruise)

    print('CL_alt =', CL_alt)
    print('CD_alt =', CD_alt)
    print('L_D_alt =', L_D_alt)
    print('C_alt =', C_altcruise)

    print('L_D_max =', L_D_max)
    print('C_loiter =', C_loiter)

    # 2 - MTOW x AR_w
    ar_w = np.arange(6.0, 14.0,0.01) # varia o alongamento de 6 a 14 no passo de 0.01 e armazena no dicionário
    m.plot_W0_x_ar_w(ar_w, airplane, 1)