# -*- coding: utf-8 -*-
"""
predecir_experimentos.py
========================
Recorre todos los experimentos en Experimentos/, carga cada modelo,
genera predicciones sobre el test set y guarda un CSV de predicciones
dentro de cada carpeta de experimento.

Uso:
    python predecir_experimentos.py

Autor: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2025
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

# ── Detección de entorno ──
if os.path.exists('/mnt/frareuirod3'):
    PROJECT_ROOT = '/mnt/frareuirod3/DeepTTA-TFG'
elif os.path.exists('/content/drive'):
    PROJECT_ROOT = '/content/drive/MyDrive/TFM/DeepTTC-main'
else:
    PROJECT_ROOT = 'C:/Users/franc/Desktop/TFM - Actualizado'

sys.path.insert(0, PROJECT_ROOT)

from Step2_DataEncoding import DataEncoding
from model_helper import Encoder_MultipleLayers, Embeddings
from dae_encoder import DAEEncoder

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP_DIR = os.path.join(PROJECT_ROOT, 'Experimentos')

# ── Clases de red (mismas que experimento.py) ──
class transformer(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.emb = Embeddings(2586, 128, 50, 0.1)
        self.encoder = Encoder_MultipleLayers(8, 128, 512, 8, 0.1, 0.1)
    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_mask = (1.0 - ex_mask) * -10000.0
        emb = self.emb(e)
        encoded = self.encoder(emb.float(), ex_mask.float())
        return encoded[:, 0]

class MLP(nn.Sequential):
    def __init__(self):
        super().__init__()
        dims = [17737, 1024, 256, 64, 256]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
    def forward(self, v):
        v = v.float().to(device)
        for l in self.predictor:
            v = F.relu(l(v))
        return v

class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_gene):
        super().__init__()
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout = nn.Dropout(0.1)
        dims = [384, 1024, 1024, 512, 1]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
    def forward(self, v_D, v_P):
        v_P = v_P.float().to(device)
        v_D = self.model_drug(v_D)
        v_P = self.model_gene(v_P)
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == len(self.predictor) - 1:
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


def build_model(cfg):
    """Construye el modelo según la config."""
    model_drug = transformer()
    if cfg['model_type'] == 'DAE':
        model_gene = DAEEncoder(
            input_dim=17737,
            latent_dim=cfg['dae_latent_dim'],
            hidden_dims=cfg['dae_hidden_dims'],
            dropout=cfg['dae_dropout'],
        )
    else:
        model_gene = MLP()
    return Classifier(model_drug, model_gene)


def predict_experiment(exp_path):
    """Genera predicciones para un experimento."""
    name = os.path.basename(exp_path)
    pred_file = os.path.join(exp_path, 'predicciones.csv')

    # Saltar si ya tiene predicciones
    if os.path.exists(pred_file):
        print(f"  ⏭  Ya existe predicciones.csv → saltando")
        return pred_file

    # Cargar config
    cfg_path = os.path.join(exp_path, 'config.json')
    if not os.path.exists(cfg_path):
        print(f"  ⚠  Sin config.json → saltando")
        return None
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    # Cargar best_model
    model_path = os.path.join(exp_path, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"  ⚠  Sin best_model.pt → saltando")
        return None

    # Preparar datos con el mismo split
    obj = DataEncoding(vocab_dir=PROJECT_ROOT)
    split_fn = getattr(obj.Getdata, cfg['data_split'])
    traindata, testdata = split_fn(random_seed=cfg['random_seed'])
    traindata, train_rna, testdata, test_rna = obj.encode(
        traindata=traindata, testdata=testdata)

    # Construir modelo y cargar pesos
    model = build_model(cfg)

    # Cargar scaler si es DAE
    scaler = None
    if cfg['model_type'] == 'DAE':
        scaler_path = os.path.join(exp_path, 'rna_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Predecir
    y_label, y_pred = [], []
    from torch.utils.data import Dataset, DataLoader

    class SimpleLoader(Dataset):
        def __init__(self, drug_df, rna_df, scaler=None):
            self.drug_df = drug_df
            self.rna_df = rna_df
            self.scaler = scaler
        def __len__(self):
            return len(self.drug_df)
        def __getitem__(self, idx):
            v_d = self.drug_df.iloc[idx]['drug_encoding']
            v_p = np.array(self.rna_df.iloc[idx])
            if self.scaler is not None:
                v_p = self.scaler.transform(v_p.reshape(1, -1)).flatten()
            y = self.drug_df.iloc[idx]['Label']
            return v_d, v_p, y

    loader = DataLoader(SimpleLoader(testdata, test_rna, scaler),
                        batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for v_d, v_p, label in loader:
            v_p = v_p.float().to(device)
            score = model(v_d, v_p)
            y_label += label.numpy().flatten().tolist()
            y_pred += torch.squeeze(score).detach().cpu().numpy().flatten().tolist()

    # Métricas
    mse = mean_squared_error(y_label, y_pred)
    rmse = np.sqrt(mse)
    pear, _ = pearsonr(y_label, y_pred)
    spear, _ = spearmanr(y_label, y_pred)
    ci = concordance_index(y_label, y_pred)

    print(f"  MSE={mse:.4f} | RMSE={rmse:.4f} | Pearson={pear:.4f} | "
          f"Spearman={spear:.4f} | CI={ci:.4f}")

    # Guardar CSV
    resultados = testdata[['DRUG_ID', 'COSMIC_ID']].copy()
    if 'TCGA_DESC' in testdata.columns:
        resultados['TCGA_DESC'] = testdata['TCGA_DESC'].values
    resultados['Label'] = y_label
    resultados['Pred'] = y_pred
    resultados.to_csv(pred_file, index=False)
    print(f"  ✅ Guardado: {pred_file}")
    return pred_file


if __name__ == '__main__':
    print("=" * 70)
    print("GENERANDO PREDICCIONES PARA TODOS LOS EXPERIMENTOS")
    print("=" * 70)

    experiments = sorted([
        d for d in os.listdir(EXP_DIR)
        if os.path.isdir(os.path.join(EXP_DIR, d)) and d.startswith('exp_')
    ])

    print(f"Encontrados {len(experiments)} experimentos\n")

    for exp_name in experiments:
        exp_path = os.path.join(EXP_DIR, exp_name)
        print(f"\n{'─'*60}")
        print(f"📁 {exp_name}")
        predict_experiment(exp_path)

    print(f"\n{'='*70}")
    print("PREDICCIONES COMPLETADAS")
    print(f"{'='*70}")
