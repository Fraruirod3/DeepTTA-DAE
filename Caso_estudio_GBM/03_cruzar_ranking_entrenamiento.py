import sys
sys.path.insert(0, '/mnt/frareuirod3/DeepTTA-TFG')

from Step1_getData import GetData
import pandas as pd

# ============================================================
# 1. Cargar los datos tal como se cargan para entrenamiento real
#    (da igual qué split usemos, los 154 fármacos son los mismos
#    en ambos: ByCancer y Random95 parten del mismo conjunto filtrado)
# ============================================================
getdata = GetData()
traindata, testdata = getdata.ByCancer(random_seed=1)

# Unir train + test para tener el universo COMPLETO de fármacos usados
todos_los_datos = pd.concat([traindata, testdata])
drugs_entrenamiento = sorted(todos_los_datos['DRUG_ID'].unique())

print(f"Fármacos únicos usados en entrenamiento+test: {len(drugs_entrenamiento)}")
print(f"Ejemplo de IDs: {drugs_entrenamiento[:10]}")

# ============================================================
# 2. Cargar tu ranking ya generado y cruzarlo
# ============================================================
ranking = pd.read_csv("ranking_farmacos_gbm_malignant.csv")

ranking['en_entrenamiento'] = ranking['DRUG_ID'].isin(drugs_entrenamiento)

print("\n=== Top 10 con indicador de si estuvo en entrenamiento ===")
print(ranking.head(10).to_string())

# ============================================================
# 3. Resumen global: cuántos del ranking completo SÍ estaban
# ============================================================
n_en_train = ranking['en_entrenamiento'].sum()
n_total = len(ranking)
print(f"\nDel total de {n_total} fármacos del ranking, "
      f"{n_en_train} ({n_en_train/n_total*100:.1f}%) estaban en el conjunto de entrenamiento.")

# Guardar versión enriquecida
ranking.to_csv("ranking_farmacos_gbm_malignant_verificado.csv", index=False)
print("\nGuardado en: ranking_farmacos_gbm_malignant_verificado.csv")