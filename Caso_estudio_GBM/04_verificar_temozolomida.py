"""
04_verificar_temozolomida.py

Verifica el hallazgo del caso de estudio de glioblastoma: por qué el modelo
predice a la temozolomida (DRUG_ID 1375) como el fármaco MENOS sensible de
todo el catálogo, pese a ser el tratamiento estándar en glioblastoma.

Compara el valor real de IC50 (Label) de temozolomida frente a la media de
todos los demás fármacos, usando el conjunto de test COMPLETO del TFM
, para comprobar si el IC50 alto de temozolomida es un patrón sistemático del propio dataset, no
específico de este perfil de paciente.

Resultado usado en la memoria: Capítulo de Resultados, sección "Un resultado
a destacar: temozolomida".
"""

import pandas as pd

# Usamos el experimento DAE Random95 (el mejor modelo, noise=0.1, ReduceLR),
# cuyo predicciones.csv contiene el Label real sobre el dataset COMPLETO
EXP_PATH = "E:\\TFM - Actualizado\\Experimentos\\exp_DAE_Random95_pretrain100_noise0.1_drop0.3_ReduceLROnPlateau_20260509_205129\\predicciones.csv"
df = pd.read_csv(EXP_PATH)

tmz = df[df['DRUG_ID'] == 1375]
print(f"Instancias de Temozolomida en el test set completo: {len(tmz)}")
print(f"\nDistribución del Label real (todas las líneas celulares):")
print(tmz['Label'].describe())

print(f"\nComparación: Label medio de TODOS los fármacos en el test set:")
print(df['Label'].describe())