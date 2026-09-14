import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, '/mnt/frareuirod3/DeepTTA-TFG')

from Step3_model_DAE import DeepTTC_DAE, data_process_loader
from Step2_DataEncoding import DataEncoding

EXP_DIR = "/mnt/frareuirod3/DeepTTA-TFG/Experimentos/exp_DAE_Random95_pretrain100_noise0.1_drop0.3_ReduceLROnPlateau_20260509_205129"
VOCAB_DIR = "/mnt/frareuirod3/DeepTTA-TFG"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Cargar el vector GBM ya construido (17737 dim, SIN normalizar)
# ============================================================
vector_gbm = np.load("gbm_malignant_vector_imputado.npy")
print(f"Vector GBM cargado: {vector_gbm.shape}")

# ============================================================
# 2. Reconstruir la arquitectura y cargar los pesos entrenados
# ============================================================
net = DeepTTC_DAE(
    modeldir=EXP_DIR,
    use_pretrained_dae=False,
    dae_hidden_dims=[1024, 512],
    dae_latent_dim=256,
    dae_dropout=0.3
)

best_model_path = os.path.join(EXP_DIR, "best_model.pt")
state_dict = torch.load(best_model_path, map_location=DEVICE)
net.model.load_state_dict(state_dict)
net.model.to(DEVICE)
net.model.eval()
print(f"Modelo cargado desde: {best_model_path}")

scaler_path = os.path.join(EXP_DIR, "rna_scaler.pkl")
with open(scaler_path, "rb") as f:
    net.rna_scaler = pickle.load(f)
print(f"Scaler cargado desde: {scaler_path}")

# ============================================================
# 3. Cargar fármacos, DETECTAR y ELIMINAR duplicados por drug_id
# ============================================================
encoder = DataEncoding(vocab_dir=VOCAB_DIR)
drug_smiles = encoder.Getdata.getDrug()

print(f"\nColumnas disponibles en getDrug(): {list(drug_smiles.columns)}")
print(f"Fármacos totales (con duplicados): {len(drug_smiles)}")

duplicados = drug_smiles[drug_smiles.duplicated(subset=['drug_id'], keep=False)]
if len(duplicados) > 0:
    print(f"\n⚠️  Se han detectado {len(duplicados)} filas duplicadas "
          f"({duplicados['drug_id'].nunique()} drug_id distintos repetidos).")
else:
    print("\n✓ No hay duplicados de drug_id.")

# Deduplicar: nos quedamos con la primera aparición de cada drug_id
drug_smiles = drug_smiles.drop_duplicates(subset=['drug_id'], keep='first').reset_index(drop=True)
print(f"Fármacos tras deduplicar: {len(drug_smiles)}")

# ============================================================
# 4. Localizar el nombre del fármaco (detector insensible a mayúsculas)
# ============================================================
posibles_columnas_nombre = ['DRUG_NAME', 'drug_name', 'Drug_Name', 'Name', 'name', 'NAME']
col_nombre = next((c for c in posibles_columnas_nombre if c in drug_smiles.columns), None)

if col_nombre is None:
    print("\n⚠️  No se ha encontrado columna de nombre de fármaco en getDrug().")
else:
    print(f"\n✓ Columna de nombre de fármaco encontrada: '{col_nombre}'")

# ============================================================
# 5. Codificar SMILES con ESPF (mismo pipeline que en entrenamiento)
# ============================================================
drug_df = drug_smiles.copy().reset_index(drop=True)
drug_df['drug_encoding'] = drug_df['smiles'].apply(encoder._drug2emb_encoder)
drug_df['Label'] = 0.0  # dummy, no se usa en inferencia pura

n_drugs = len(drug_df)
print(f"\nFármacos codificados: {n_drugs}")

# ============================================================
# 6. Repetir el vector GBM una vez por fármaco
# ============================================================
rna_df = pd.DataFrame(np.tile(vector_gbm, (n_drugs, 1)))
rna_df.index = range(n_drugs)

# ============================================================
# 7. Inferencia pura (sin calcular métricas, no hay verdad real)
# ============================================================
info = data_process_loader(
    drug_df.index.values,
    drug_df['Label'].values,
    drug_df, rna_df,
    scaler=net.rna_scaler
)
params = {'batch_size': 16, 'shuffle': False, 'num_workers': 0,
          'drop_last': False, 'sampler': SequentialSampler(info)}
generator = DataLoader(info, **params)

y_pred = []
with torch.no_grad():
    for v_drug, v_gene, _ in generator:
        v_gene = v_gene.float().to(DEVICE)
        score = net.model(v_drug, v_gene)
        y_pred += torch.squeeze(score).detach().cpu().numpy().flatten().tolist()

print(f"Predicciones generadas: {len(y_pred)}")

# ============================================================
# 8. Ranking final, con nombre de fármaco si está disponible
# ============================================================
cols_resultado = {"DRUG_ID": drug_df['drug_id'].values, "ln_IC50_pred": y_pred}
if col_nombre is not None:
    cols_resultado["DRUG_NAME"] = drug_df[col_nombre].values
if 'Targets' in drug_df.columns:
    cols_resultado["Targets"] = drug_df['Targets'].values
if 'Target pathway' in drug_df.columns:
    cols_resultado["Target_pathway"] = drug_df['Target pathway'].values

resultados = pd.DataFrame(cols_resultado).sort_values("ln_IC50_pred").reset_index(drop=True)

resultados.to_csv("ranking_farmacos_gbm_malignant.csv", index=False)
print("\nTop 10 fármacos más sensibles (menor IC50 predicho):")
print(resultados.head(10).to_string())
print("\nGuardado en: ranking_farmacos_gbm_malignant.csv")

# ============================================================
# 9. Identificación detallada del Top 10 + comprobar SMILES compartido
#    entre valores empatados exactos (posible mismo compuesto con
#    distinto drug_id, p.ej. registrado en GDSC1 y GDSC2 por separado)
# ============================================================
top10_ids = resultados.head(10)['DRUG_ID'].tolist()

info_top10 = drug_smiles[drug_smiles['drug_id'].isin(top10_ids)][
    ['drug_id', col_nombre if col_nombre else 'drug_id', 'Targets', 'Target pathway', 'smiles']
].drop_duplicates(subset=['drug_id'])

print("\n=== Identificación detallada del Top 10 ===")
print(info_top10.to_string())

# Detectar pares de IDs distintos con predicción idéntica (posible SMILES compartido)
preds_top10 = resultados.head(10)
duplicados_pred = preds_top10[preds_top10.duplicated(subset=['ln_IC50_pred'], keep=False)]

if len(duplicados_pred) > 0:
    print("\n=== IDs con predicción idéntica (posible SMILES compartido) ===")
    for valor in duplicados_pred['ln_IC50_pred'].unique():
        ids_empatados = duplicados_pred[duplicados_pred['ln_IC50_pred'] == valor]['DRUG_ID'].tolist()
        smiles_pair = drug_smiles[drug_smiles['drug_id'].isin(ids_empatados)][['drug_id', 'smiles']]
        print(f"\nPredicción {valor:.6f} -> drug_ids {ids_empatados}:")
        print(smiles_pair.to_string())
        if smiles_pair['smiles'].nunique() == 1:
            print("  ✓ Confirmado: mismo SMILES exacto -> es el mismo compuesto químico.")
        else:
            print("  ⚠️ SMILES distintos -> la coincidencia de predicción es casual, no por ser el mismo compuesto.")
else:
    print("\n✓ No hay predicciones empatadas en el Top 10.")