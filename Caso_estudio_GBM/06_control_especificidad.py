"""
06_control_especificidad.py

Prueba de control para la validación cruzada del caso de estudio: repite el
mismo análisis de "top 10 por IC50 real" pero sobre un tipo de cáncer
distinto a GBM (con biología muy diferente), para comprobar si la
coincidencia de Bortezomib/Dactinomicina es específica de glioblastoma o
simplemente refleja fármacos citotóxicos generales en cualquier tipo tumoral.
"""

import pandas as pd
import sys, os

if os.path.exists('/mnt/frareuirod3'):
    sys.path.insert(0, '/mnt/frareuirod3/DeepTTA-TFG')
else:
    sys.path.insert(0, 'E:/TFM - Actualizado')

from Step1_getData import GetData

getdata = GetData()
drug_cell_df = pd.read_excel(getdata.pairfile)
drug_cell_df = getdata._filter_pair(drug_cell_df)

# Cambia aquí el tipo de cáncer a comparar
TIPO_CONTROL = 'LAML'  # Leucemia mieloide aguda - biología muy distinta a GBM

control = drug_cell_df[drug_cell_df['TCGA_DESC'] == TIPO_CONTROL]
print(f"Instancias {TIPO_CONTROL} en GDSC2 completo: {len(control)}")

agg_label = control.groupby('DRUG_ID').agg(
    n_lineas=('COSMIC_ID', 'nunique'),
    LN_IC50_medio=('LN_IC50', 'mean')
).sort_values('LN_IC50_medio')

print(f"\n=== TOP 10 por LN_IC50 real en {TIPO_CONTROL} ===")
print(agg_label.head(10).to_string())