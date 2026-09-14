# python3
# -*- coding: utf-8 -*-

"""
experimento.py - Lanzador de experimentos documentados para el TFM
===================================================================

Soporta dos tipos de modelo:
    "model_type": "MLP"  →  Transformer + MLP (modelo original DeepTTA)
    "model_type": "DAE"  →  Transformer + DAE (aportación del TFM)

Cada ejecución crea una carpeta en Experimentos/ con:
    - config.json               → hiperparámetros exactos
    - valid_markdowntable.txt   → métricas por época
    - loss_curve_iter.pkl       → historial de pérdida
    - best_model.pt             → mejor checkpoint (menor MSE validación)
    - log.txt                   → salida completa del entrenamiento
    [Solo DAE]:
    - dae_pretrained.pt         → pesos del DAE tras preentrenamiento
    - rna_scaler.pkl            → scaler de normalización

Para comparar todos los experimentos:
    python comparar_experimentos.py

Autor: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2025
"""

import os
import json
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from prettytable import PrettyTable

from model_helper import Encoder_MultipleLayers, Embeddings
from dae_encoder import DAEEncoder

# ============================================================================
# CONFIGURACIÓN DEL EXPERIMENTO — edita solo este bloque entre ejecuciones
# ============================================================================
CONFIG = {
    "descripcion":  "MLP 100k ByCancer con ReduceLROnPlateau",
    "model_type":   "MLP",
    "data_split":   "ByCancer",
    "random_seed":  1,

    "pretrain_epochs":  100,
    "pretrain_lr":      1e-3,
    "noise_factor":     0.3,
    "pretrain_batch":   64,

    "dae_hidden_dims":  [1024, 512],
    "dae_latent_dim":   256,
    "dae_dropout":      0.3,

    "train_epochs":     200,
    "train_lr":         1e-4,
    "weight_decay":     0,
    "batch_size":       64,

    "scheduler":            "ReduceLROnPlateau",
    "scheduler_patience":   10,
    "scheduler_factor":     0.5,
    "scheduler_T_max":      200,
}




# ============================================================================
# DETECCIÓN DE ENTORNO Y RUTAS
# ============================================================================
if os.path.exists('/content/drive'):
    vocab_dir   = '/content/drive/MyDrive/TFM/DeepTTC-main'
    base_expdir = '/content/drive/MyDrive/TFM/DeepTTA-TFG/Experimentos'
elif os.path.exists('/mnt/frareuirod3'):
    vocab_dir   = '/mnt/frareuirod3/DeepTTA-TFG'
    base_expdir = '/mnt/frareuirod3/DeepTTA-TFG/Experimentos'
else:
    vocab_dir   = 'C:/Users/franc/Desktop/TFG/DeepTTC-main'
    base_expdir = 'C:/Users/franc/Desktop/TFM-DeepTTA/Experimentos'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[experimento.py] Dispositivo: {device}")


# ============================================================================
# NOMBRE AUTOMÁTICO DEL EXPERIMENTO
# ============================================================================
def nombre_experimento(cfg):
    scheduler_str = cfg['scheduler'] if cfg['scheduler'] else 'NoScheduler'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg['model_type'] == 'DAE':
        return (
            f"exp_DAE"
            f"_{cfg['data_split']}"
            f"_pretrain{cfg['pretrain_epochs']}"
            f"_noise{cfg['noise_factor']}"
            f"_drop{cfg['dae_dropout']}"
            f"_{scheduler_str}"
            f"_{timestamp}"
        )
    else:
        return (
            f"exp_MLP"
            f"_{cfg['data_split']}"
            f"_{scheduler_str}"
            f"_{timestamp}"
        )


# ============================================================================
# DATA LOADER — con soporte opcional de scaler (DAE lo usa, MLP no)
# ============================================================================
class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, rna_df, scaler=None):
        self.labels   = labels
        self.list_IDs = list_IDs
        self.drug_df  = drug_df
        self.rna_df   = rna_df
        self.scaler   = scaler

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d   = self.drug_df.iloc[index]['drug_encoding']
        v_p   = np.array(self.rna_df.iloc[index])
        if self.scaler is not None:
            v_p = self.scaler.transform(v_p.reshape(1, -1)).flatten()
        y = self.labels[index]
        return v_d, v_p, y


# ============================================================================
# COMPONENTES DE RED — Transformer (compartido) + MLP + Classifier
# ============================================================================
class transformer(nn.Sequential):
    def __init__(self):
        super(transformer, self).__init__()
        self.emb = Embeddings(2586, 128, 50, 0.1)
        self.encoder = Encoder_MultipleLayers(8, 128, 512, 8, 0.1, 0.1)

    def forward(self, v):
        e        = v[0].long().to(device)
        e_mask   = v[1].long().to(device)
        ex_mask  = e_mask.unsqueeze(1).unsqueeze(2)
        ex_mask  = (1.0 - ex_mask) * -10000.0
        emb      = self.emb(e)
        encoded  = self.encoder(emb.float(), ex_mask.float())
        return encoded[:, 0]


class MLP(nn.Sequential):
    def __init__(self):
        super(MLP, self).__init__()
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
        super(Classifier, self).__init__()
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout    = nn.Dropout(0.1)
        dims = [384, 1024, 1024, 512, 1]
        self.predictor  = nn.ModuleList(
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


# ============================================================================
# CLASE EXPERIMENTO
# ============================================================================
class Experimento:
    """
    Gestiona un experimento completo: construcción del modelo, entrenamiento,
    guardado de pesos, config y métricas.
    """

    def __init__(self, config, expdir):
        self.cfg    = config
        self.expdir = expdir
        self.scaler = None

        # Construir modelo según model_type
        model_drug = transformer()

        if config['model_type'] == 'DAE':
            model_gene = DAEEncoder(
                input_dim   = 17737,
                latent_dim  = config['dae_latent_dim'],
                hidden_dims = config['dae_hidden_dims'],
                dropout     = config['dae_dropout'],
            )
        else:
            model_gene = MLP()

        self.model = Classifier(model_drug, model_gene)

    # -------------------------------------------------------------------------
    def guardar_config(self):
        path = os.path.join(self.expdir, 'config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.cfg, f, indent=4, ensure_ascii=False)
        print(f"Config guardada en: {path}")

    # -------------------------------------------------------------------------
    def pretrain_dae(self, train_rna):
        """Fase 1: preentrenamiento del DAE. Solo se ejecuta si model_type=DAE."""
        print("=" * 70)
        print("FASE 1: PREENTRENAMIENTO DEL DAE")
        print("=" * 70)

        rna_array = train_rna.values.astype(np.float32)
        print(f"  Rango original: [{rna_array.min():.2f}, {rna_array.max():.2f}]")

        self.scaler = MinMaxScaler()
        rna_norm    = self.scaler.fit_transform(rna_array)
        print(f"  Rango normalizado: [{rna_norm.min():.2f}, {rna_norm.max():.2f}]")

        # Guardar scaler
        scaler_path = os.path.join(self.expdir, 'rna_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        rna_tensor  = torch.FloatTensor(rna_norm)
        dataset     = TensorDataset(rna_tensor)
        dataloader  = DataLoader(dataset, batch_size=self.cfg['pretrain_batch'], shuffle=True)

        dae = self.model.model_gene
        dae.pretrain(
            data_loader   = dataloader,
            epochs        = self.cfg['pretrain_epochs'],
            noise_factor  = self.cfg['noise_factor'],
            learning_rate = self.cfg['pretrain_lr'],
            device        = device,
        )

        dae_path = os.path.join(self.expdir, 'dae_pretrained.pt')
        dae.save_pretrained(dae_path)

    # -------------------------------------------------------------------------
    def _build_scheduler(self, opt):
        s = self.cfg['scheduler']
        if s == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min',
                patience = self.cfg['scheduler_patience'],
                factor   = self.cfg['scheduler_factor'],
            )
        elif s == 'CosineAnnealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg['scheduler_T_max']
            )
        return None

    # -------------------------------------------------------------------------
    def _test(self, datagenerator):
        y_label, y_pred = [], []
        self.model.eval()
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            for v_d, v_p, label in datagenerator:
                v_p   = v_p.float().to(device)
                score = self.model(v_d, v_p)
                n     = torch.squeeze(score, 1)
                loss  = loss_fn(n, torch.from_numpy(np.array(label)).float().to(device))
                y_label += label.numpy().flatten().tolist()
                y_pred  += torch.squeeze(score).detach().cpu().numpy().flatten().tolist()

        self.model.train()
        mse      = mean_squared_error(y_label, y_pred)
        rmse     = np.sqrt(mse)
        pearson, p_val   = pearsonr(y_label, y_pred)
        spearman, s_pval = spearmanr(y_label, y_pred)
        ci               = concordance_index(y_label, y_pred)
        return mse, rmse, pearson, p_val, spearman, s_pval, ci, loss

    # -------------------------------------------------------------------------
    def train(self, train_drug, train_rna, val_drug, val_rna):
        cfg  = self.cfg
        opt  = torch.optim.Adam(
            self.model.parameters(),
            lr           = cfg['train_lr'],
            weight_decay = cfg['weight_decay'],
        )
        scheduler = self._build_scheduler(opt)

        self.model = self.model.to(device)

        params = {'batch_size': cfg['batch_size'], 'shuffle': True,
                  'num_workers': 0, 'drop_last': False}

        train_loader = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values,
            train_drug, train_rna, scaler=self.scaler), **params)
        val_loader   = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values,
            val_drug, val_rna, scaler=self.scaler), **params)

        loss_fn      = nn.MSELoss()
        loss_history = []
        max_MSE      = 10000
        model_max    = copy.deepcopy(self.model)

        header = ['# epoch', 'MSE', 'RMSE', 'Pearson Correlation', 'with p-value',
                  'Spearman Correlation', 'with p-value2', 'Concordance Index', 'LR']
        table     = PrettyTable(header)
        float2str = lambda x: '%0.4f' % x
        t_start   = time.time()

        print('=' * 70)
        print(f"ENTRENAMIENTO: {cfg['model_type']} | {cfg['data_split']} | scheduler={cfg['scheduler']}")
        print('=' * 70)

        for epo in range(cfg['train_epochs']):
            self.model.train()
            for i, (v_d, v_p, label) in enumerate(train_loader):
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label))).float().to(device)
                loss  = loss_fn(torch.squeeze(score, 1).float(), label)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % 1000 == 0:
                    elapsed = (time.time() - t_start) / 3600
                    print(f'  Epoch {epo+1} | iter {i} | loss {loss.item():.5f} | {elapsed:.3f}h')

            # Validación
            mse, rmse, pearson, p_val, spearman, s_pval, ci, _ = self._test(val_loader)
            current_lr = opt.param_groups[0]['lr']

            # Step scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mse)
            elif scheduler is not None:
                scheduler.step()

            # Guardar mejor modelo
            if mse < max_MSE:
                model_max  = copy.deepcopy(self.model)
                max_MSE    = mse
                torch.save(self.model.state_dict(),
                           os.path.join(self.expdir, 'best_model.pt'))
                print(f'\n  MEJOR (epoch {epo+1}): MSE={mse:.4f} | Pearson={pearson:.4f} | LR={current_lr:.2e}\n')

            row = [f'epoch {epo}'] + list(map(float2str,
                  [mse, rmse, pearson, p_val, spearman, s_pval, ci])) + [f'{current_lr:.2e}']
            table.add_row(row)

        self.model = model_max

        with open(os.path.join(self.expdir, 'valid_markdowntable.txt'), 'w') as f:
            f.write(table.get_string())
        with open(os.path.join(self.expdir, 'loss_curve_iter.pkl'), 'wb') as f:
            pickle.dump(loss_history, f)

        print('--- Entrenamiento finalizado ---')

    # -------------------------------------------------------------------------
    def guardar_modelo(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.expdir, 'model.pt'))
        print(f"Modelo final guardado en: {self.expdir}/model.pt")


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================
if __name__ == '__main__':

    # Crear carpeta del experimento
    exp_name = nombre_experimento(CONFIG)
    expdir   = os.path.join(base_expdir, exp_name)
    os.makedirs(expdir, exist_ok=True)

    # Redirigir stdout a log.txt además de la consola
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file   = open(os.path.join(expdir, 'log.txt'), 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 70)
    print(f"EXPERIMENTO: {exp_name}")
    print(f"Descripción: {CONFIG['descripcion']}")
    print(f"Modelo:      {CONFIG['model_type']}")
    print(f"Dispositivo: {device}")
    print(f"Carpeta:     {expdir}")
    print("=" * 70)

    # Cargar datos
    from Step2_DataEncoding import DataEncoding

    obj      = DataEncoding(vocab_dir=vocab_dir)
    split_fn = getattr(obj.Getdata, CONFIG['data_split'])
    traindata, testdata = split_fn(random_seed=CONFIG['random_seed'])
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata, testdata=testdata)

    print(f"\nDatos:")
    print(f"  Split:  {CONFIG['data_split']}")
    print(f"  Train:  {len(traindata)} muestras")
    print(f"  Test:   {len(testdata)} muestras")
    print(f"  Genes:  {train_rnadata.shape[1]}\n")

    # Crear experimento y guardar config
    exp = Experimento(CONFIG, expdir)
    exp.guardar_config()

    # Fase 1: preentrenamiento DAE (solo si model_type=DAE)
    if CONFIG['model_type'] == 'DAE':
        exp.pretrain_dae(train_rnadata)

    # Fase 2: entrenamiento conjunto
    exp.train(traindata, train_rnadata, testdata, test_rnadata)

    # Guardar modelo final
    exp.guardar_modelo()

    print("=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print(f"Resultados en: {expdir}")
    print("=" * 70)

    log_file.close()
