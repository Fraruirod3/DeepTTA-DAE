import pandas as pd
import sys
import os

# Añadir la carpeta raíz del proyecto al path, para poder importar Step1_getData
# (funciona tanto en local como en servidor)
if os.path.exists('/mnt/frareuirod3'):
    sys.path.insert(0, '/mnt/frareuirod3/DeepTTA-TFG')
else:
    sys.path.insert(0, 'E:/TFM - Actualizado')

from Step1_getData import GetData

getdata = GetData()

drug_cell_df = pd.read_excel(getdata.pairfile)
drug_cell_df = getdata._filter_pair(drug_cell_df)

gbm = drug_cell_df[drug_cell_df['TCGA_DESC'] == 'GBM']
print(f"Instancias GBM en GDSC2 completo (tras filtrado): {len(gbm)}")

agg_label = gbm.groupby('DRUG_ID').agg(
    n_lineas=('COSMIC_ID', 'nunique'),
    LN_IC50_medio=('LN_IC50', 'mean')
).sort_values('LN_IC50_medio')

print("\n=== TOP 10 por LN_IC50 real (todas las instancias GBM de GDSC2) ===")
print(agg_label.head(10).to_string())