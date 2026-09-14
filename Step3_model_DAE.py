# python3
# -*- coding:utf-8 -*-

"""
Step3_model_DAE.py - DeepTTA con DAE en lugar de MLP
=====================================================

Este archivo es una versión modificada de Step3_model.py que reemplaza
el MLP simple por un Denoising Autoencoder (DAE) preentrenado.

DIFERENCIAS CON EL ORIGINAL:
-----------------------------
1. Importa DAEEncoder desde dae_encoder.py
2. La clase DeepTTC_DAE tiene un paso adicional de PREENTRENAMIENTO
3. El encoder de genes (DAE) se preentrena ANTES del entrenamiento conjunto

¿CÓMO FUNCIONA?
---------------
ORIGINAL (MLP):
    Datos → MLP → Representación (256-dim) → Classifier → IC50
    (Todo se entrena junto, el MLP aprende "en frío")

NUEVO (DAE):
    FASE 1 - Preentrenamiento:
        Datos → DAE → Reconstrucción de datos
        (El DAE aprende la estructura de los datos de expresión)
    
    FASE 2 - Entrenamiento conjunto:
        Datos → DAE (preentrenado) → Representación (256-dim) → Classifier → IC50
        (El DAE ya tiene buenos pesos iniciales)

Autor original: 野山羊骑士
Modificado por: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2025
"""

import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset

# TensorBoard deshabilitado debido a conflictos con TensorFlow/NumPy
# Si quieres usar TensorBoard, arregla la instalación de TensorFlow primero
TENSORBOARD_AVAILABLE = False
SummaryWriter = None  # Placeholder

from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from model_helper import Encoder_MultipleLayers, Embeddings

# ============================================================================
# NUEVA IMPORTACIÓN: DAE Encoder
# ============================================================================
from dae_encoder import DAEEncoder

# Detectar GPU automáticamente
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DeepTTA-DAE] Dispositivo: {device}")


# ============================================================================
# DATA LOADER (modificado para soportar normalización)
# ============================================================================
class data_process_loader(data.Dataset):
    """
    Dataset loader para DeepTTA.
    Carga pares de (codificación del fármaco, expresión génica, etiqueta IC50).
    
    Modificado para soportar normalización de datos de expresión génica.
    """
    def __init__(self, list_IDs, labels, drug_df, rna_df, scaler=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df
        self.rna_df = rna_df
        self.scaler = scaler  # MinMaxScaler para normalizar datos

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.drug_df.iloc[index]['drug_encoding']
        v_p = np.array(self.rna_df.iloc[index])
        
        # Normalizar si hay scaler disponible
        if self.scaler is not None:
            v_p = self.scaler.transform(v_p.reshape(1, -1)).flatten()
        
        y = self.labels[index]
        return v_d, v_p, y


# ============================================================================
# TRANSFORMER (sin cambios - codifica SMILES del fármaco)
# ============================================================================
class transformer(nn.Sequential):
    """
    Transformer para codificar la estructura molecular del fármaco (SMILES).
    
    Input: Secuencia SMILES codificada con BPE (máximo 50 tokens)
    Output: Vector de 128 dimensiones
    
    Este componente NO se modifica, solo cambiamos el encoder de genes.
    """
    def __init__(self):
        super(transformer, self).__init__()
        input_dim_drug = 2586           # Tamaño del vocabulario ESPF
        transformer_emb_size_drug = 128  # Dimensión de embedding
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8     # Número de capas del transformer
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,  # Máxima longitud de secuencia
                         transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    
    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]  # CLS token (primera posición)


# ============================================================================
# MLP ORIGINAL (mantenido para comparación)
# ============================================================================
class MLP(nn.Sequential):
    """
    MLP original para codificar expresión génica.
    
    MANTENENEMOS ESTA CLASE para poder comparar resultados con el DAE.
    
    Arquitectura:
        17737 genes → 1024 → 256 → 64 → 256 (output)
    """
    def __init__(self):
        input_dim_gene = 17737
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 256, 64]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v


# ============================================================================
# CLASSIFIER (modificado para aceptar DAE o MLP)
# ============================================================================
class Classifier(nn.Sequential):
    """
    Clasificador que combina embeddings de fármaco y genes para predecir IC50.
    
    Arquitectura:
        [drug_emb (128) || gene_emb (256)] → 1024 → 1024 → 512 → 1 (IC50)
    
    Este clasificador funciona igual tanto con MLP como con DAE,
    ya que ambos producen un vector de 256 dimensiones.
    """
    def __init__(self, model_drug, model_gene):
        super(Classifier, self).__init__()
        self.input_dim_drug = 128   # Output del transformer
        self.input_dim_gene = 256   # Output del MLP o DAE
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout = nn.Dropout(0.1)
        self.hidden_dims = [1024, 1024, 512]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_gene] + self.hidden_dims + [1]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # ⚠️ SOLUCIÓN: Enviar datos de genes a la GPU antes de procesarlos
        v_P = v_P.float().to(device)
        # Encoding de cada modalidad
        v_D = self.model_drug(v_D)  # Fármaco → 128 dim
        v_P = self.model_gene(v_P)  # Genes → 256 dim
        
        # Concatenar y clasificar
        v_f = torch.cat((v_D, v_P), 1)  # → 384 dim
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)  # Última capa sin activación
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


# ============================================================================
# CLASE PRINCIPAL: DeepTTC_DAE
# ============================================================================
class DeepTTC_DAE:
    """
    DeepTTA con Denoising Autoencoder (DAE) para codificación de genes.
    
    DIFERENCIA PRINCIPAL con DeepTTC original:
    ------------------------------------------
    1. Usa DAEEncoder en lugar de MLP para procesar expresión génica
    2. Tiene una fase de PREENTRENAMIENTO del DAE antes del entrenamiento conjunto
    3. El DAE se preentrena con ruido para aprender representaciones robustas
    
    Flujo de entrenamiento:
    1. pretrain_dae() - Preentrenar el DAE con los datos de expresión
    2. train() - Entrenar el modelo completo (igual que el original)
    
    Parámetros:
    -----------
    modeldir : str
        Directorio donde guardar los modelos y logs
    
    use_pretrained_dae : bool
        Si True, intenta cargar un DAE preentrenado existente
    
    dae_hidden_dims : list
        Dimensiones de las capas ocultas del DAE (default: [1024, 512])
    
    dae_latent_dim : int
        Dimensión del espacio latente del DAE (default: 256, igual que MLP)
    """
    
    def __init__(self, modeldir, 
                 use_pretrained_dae=False,
                 dae_hidden_dims=[1024, 512],
                 dae_latent_dim=256,
                 dae_dropout=0.3):
        
        # ====================================================================
        # CREAR COMPONENTES DEL MODELO
        # ====================================================================
        
        # Transformer para fármacos (sin cambios)
        model_drug = transformer()
        
        # DAE para genes (NUEVO - reemplaza al MLP)
        model_gene = DAEEncoder(
            input_dim=17737,          # Número de genes
            latent_dim=dae_latent_dim, # Dimensión de salida (debe ser 256)
            hidden_dims=dae_hidden_dims,
            dropout=dae_dropout
        )
        
        # Clasificador que combina ambos
        self.model = Classifier(model_drug, model_gene)
        
        # Configuración
        self.device = device
        self.modeldir = modeldir
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        
        # Rutas para guardar el DAE preentrenado
        self.dae_pretrained_path = os.path.join(modeldir, "dae_pretrained.pt")
        
        # Si existe un DAE preentrenado, cargarlo
        if use_pretrained_dae and os.path.exists(self.dae_pretrained_path):
            print(f"Cargando DAE preentrenado desde: {self.dae_pretrained_path}")
            self.model.model_gene.load_pretrained(self.dae_pretrained_path, device)
    
    # ========================================================================
    # NUEVO MÉTODO: PREENTRENAMIENTO DEL DAE
    # ========================================================================
    def pretrain_dae(self, train_rna, 
                     epochs=100, 
                     batch_size=64,
                     noise_factor=0.3,
                     learning_rate=1e-3):
        """
        Preentrena el DAE usando los datos de expresión génica.
        
        Este paso es CRUCIAL y es lo que diferencia nuestro enfoque del MLP simple.
        El DAE aprende la estructura de los datos de expresión ANTES de intentar
        predecir IC50.
        
        IMPORTANTE: Los datos se normalizan a [0, 1] porque el DAE usa Sigmoid
        en la capa de salida, que solo produce valores en ese rango.
        
        Args:
            train_rna: DataFrame con datos de expresión génica
            epochs: Épocas de preentrenamiento (100 es un buen default)
            batch_size: Tamaño del batch
            noise_factor: Cantidad de ruido (0.3 = 30%)
            learning_rate: Tasa de aprendizaje
        
        Returns:
            loss_history: Lista con pérdidas por época
        """
        from sklearn.preprocessing import MinMaxScaler
        
        print("=" * 70)
        print("FASE 1: PREENTRENAMIENTO DEL DAE")
        print("=" * 70)
        print(f"  Datos de entrenamiento: {train_rna.shape}")
        print(f"  Épocas: {epochs}")
        print(f"  Factor de ruido: {noise_factor}")
        print("=" * 70)
        
        # Convertir DataFrame a array
        rna_array = train_rna.values.astype(np.float32)
        
        # ====================================================================
        # NORMALIZACIÓN: Escalar datos a rango [0, 1]
        # ====================================================================
        # El DAE usa Sigmoid en la salida, que solo produce valores en [0, 1].
        # Si los datos originales están en rango [2, 14], el MSE será enorme
        # porque el modelo nunca puede predecir valores fuera de [0, 1].
        print(f"  Rango original: [{rna_array.min():.2f}, {rna_array.max():.2f}]")
        
        self.rna_scaler = MinMaxScaler()
        rna_normalized = self.rna_scaler.fit_transform(rna_array)
        
        print(f"  Rango normalizado: [{rna_normalized.min():.2f}, {rna_normalized.max():.2f}]")
        print("=" * 70)
        
        # Convertir a Tensor
        rna_tensor = torch.FloatTensor(rna_normalized)
        
        # Crear DataLoader
        dataset = TensorDataset(rna_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Preentrenar el DAE
        dae = self.model.model_gene
        loss_history = dae.pretrain(
            data_loader=dataloader,
            epochs=epochs,
            noise_factor=noise_factor,
            learning_rate=learning_rate,
            device=self.device,
            verbose=True
        )
        
        # Guardar el DAE preentrenado
        os.makedirs(self.modeldir, exist_ok=True)
        dae.save_pretrained(self.dae_pretrained_path)
        
        # Guardar también el scaler para usarlo en inferencia
        import pickle
        scaler_path = os.path.join(self.modeldir, "rna_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.rna_scaler, f)
        print(f"Scaler guardado en: {scaler_path}")
        
        return loss_history

    # ========================================================================
    # MÉTODO TEST (sin cambios significativos)
    # ========================================================================
    def test(self, datagenerator, model):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Calcula métricas de regresión:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - Pearson Correlation
        - Spearman Correlation
        - Concordance Index
        """
        y_label = []
        y_pred = []
        model.eval()
        
        for i, (v_drug, v_gene, label) in enumerate(datagenerator):
            v_gene = v_gene.float().to(self.device)
            score = model(v_drug, v_gene)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        model.train()

        return y_label, y_pred, \
               mean_squared_error(y_label, y_pred), \
               np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               pearsonr(y_label, y_pred)[1], \
               spearmanr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[1], \
               concordance_index(y_label, y_pred), \
               loss

    # ========================================================================
    # MÉTODO TRAIN (igual que el original)
    # ========================================================================
    def train(self, train_drug, train_rna, val_drug, val_rna):
        """
        Entrena el modelo completo.
        
        IMPORTANTE: Si el DAE no ha sido preentrenado, se entrenará desde cero
        junto con el resto del modelo (como el MLP original).
        Para mejores resultados, llamar a pretrain_dae() ANTES de este método.
        """
        
        # Verificar si el DAE fue preentrenado
        if self.model.model_gene.is_pretrained:
            print("✓ El DAE ya fue preentrenado. Usando pesos aprendidos.")
        else:
            print("⚠ ADVERTENCIA: El DAE NO ha sido preentrenado.")
            print("  Para mejores resultados, llama a pretrain_dae() antes de train().")
        
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 64
        train_epoch = 200
        self.model = self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        
        # Obtener el scaler (puede ser None si no se llamó a pretrain_dae)
        scaler = getattr(self, 'rna_scaler', None)
        if scaler is not None:
            print("✓ Usando scaler para normalizar datos de expresión génica")
        
        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug, train_rna, scaler=scaler), **params)
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna, scaler=scaler), **params)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch', "MSE", 'RMSE',
                               "Pearson Correlation", "with p-value",
                               'Spearman Correlation', "with p-value2",
                               "Concordance Index"]
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x
        
        print('=' * 70)
        print('FASE 2: ENTRENAMIENTO CONJUNTO (DAE + Transformer + Classifier)')
        print('=' * 70)
        
        # TensorBoard writer (opcional)
        writer = None
        if TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_DAE')
        t_start = time.time()
        iteration_loss = 0

        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                if TENSORBOARD_AVAILABLE and writer is not None:
                    writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) +
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] +
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.set_grad_enabled(False):
                y_true, y_pred, mse, rmse, \
                person, p_val, \
                spearman, s_p_val, CI, \
                loss_val = self.test(validation_generator, self.model)
                
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)
                
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                    
                    # ============================================================
                    # GUARDAR MEJOR MODELO (checkpoint)
                    # ============================================================
                    best_model_path = os.path.join(self.modeldir, 'best_model_dae.pt')
                    torch.save(self.model.state_dict(), best_model_path)
                    
                    # Mostrar métricas completas al guardar
                    print(f'\n💾 MEJOR MODELO GUARDADO en: {best_model_path}')
                    print(f'   Validation at Epoch {epo + 1}:')
                    print(f'   - MSE: {mse:.5f}')
                    print(f'   - RMSE: {rmse:.5f}')
                    print(f'   - Pearson: {person:.5f} (p-value: {p_val:.2e})')
                    print(f'   - Spearman: {spearman:.5f} (p-value: {s_p_val:.2e})')
                    print(f'   - Concordance Index: {CI:.5f}\n')
                    
                    if TENSORBOARD_AVAILABLE and writer is not None:
                        writer.add_scalar("valid/mse", mse, epo)
                        writer.add_scalar('valida/rmse', rmse, epo)
                        writer.add_scalar("valid/pearson_correlation", person, epo)
                        writer.add_scalar("valid/concordance_index", CI, epo)
                        writer.add_scalar("valid/Spearman", spearman, epo)
                        writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
            table.add_row(lst)

        self.model = model_max

        with open(self.record_file, 'w') as fp:
            fp.write(table.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        print('--- Training Finished ---')
        if TENSORBOARD_AVAILABLE and writer is not None:
            writer.flush()
            writer.close()

    # ========================================================================
    # MÉTODO PREDICT (sin cambios)
    # ========================================================================
    def predict(self, drug_data, rna_data):
        """Realiza predicciones con el modelo entrenado."""
        print('predicting...')
        self.model.to(device)
        
        # Obtener el scaler (necesario para normalizar datos)
        scaler = getattr(self, 'rna_scaler', None)
        if scaler is None:
            print("⚠️ ADVERTENCIA: No hay scaler cargado. Los resultados pueden ser incorrectos.")
        
        info = data_process_loader(drug_data.index.values,
                                   drug_data.Label.values,
                                   drug_data, rna_data, scaler=scaler)
        params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 0,
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}
        generator = data.DataLoader(info, **params)

        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(generator, self.model)

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI

    # ========================================================================
    # MÉTODOS DE GUARDADO Y CARGA
    # ========================================================================
    def save_model(self):
        """Guarda el modelo completo."""
        torch.save(self.model.state_dict(), self.modeldir + '/model_dae.pt')
        print(f"Modelo guardado en: {self.modeldir}/model_dae.pt")

    def load_pretrained(self, path):
        """Carga un modelo previamente entrenado."""
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    print("=" * 70)
    print("DeepTTA con DAE - Entrenamiento (Modelo  - 100k muestras)")
    print("=" * 70)
    
    # ========================================================================
    # PASO 1: Cargar y preparar datos
    # ========================================================================
    from Step2_DataEncoding import DataEncoding
    
    # Detectar entorno: Colab vs Local vs Servidor
    if os.path.exists('/content/drive'):
        # Google Colab
        vocab_dir = '/content/drive/MyDrive/TFM/DeepTTC-main'
        base_modeldir = '/content/drive/MyDrive/TFM/DeepTTA-TFG/Model_DAE'
    elif os.path.exists('/mnt/frareuirod3'):
        # Servidor Linux
        vocab_dir = '/mnt/frareuirod3/DeepTTA-TFG'
        base_modeldir = '/mnt/frareuirod3/DeepTTA-TFG/Model_DAE'
    else:
        # Local Windows
        vocab_dir = 'C:/Users/franc/Desktop/TFG/DeepTTC-main'
        base_modeldir = 'C:/Users/franc/Desktop/TFM/DeepTTA-TFG/Model_DAE'
    
    obj = DataEncoding(vocab_dir=vocab_dir)

    # Dividir datos (limit=40000 para Modelo 2)
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)
    
    # Codificar datos
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)
    
    print(f"\nDatos cargados:")
    print(f"  Train: {len(traindata)} muestras")
    print(f"  Test: {len(testdata)} muestras")
    print(f"  Genes: {train_rnadata.shape[1]}")

    # ========================================================================
    # PASO 2: Crear modelo y directorio
    # ========================================================================
    modeldir = f'{base_modeldir}/Modelo_2'
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    
    # Crear modelo con DAE
    net = DeepTTC_DAE(
        modeldir=modeldir,
        dae_hidden_dims=[1024, 512],
        dae_latent_dim=256,
        dae_dropout=0.3
    )
    
    # ========================================================================
    # PASO 3: PREENTRENAR EL DAE (NUEVO PASO)
    # ========================================================================
    net.pretrain_dae(
        train_rna=train_rnadata,
        epochs=100,          # Épocas de preentrenamiento
        batch_size=64,
        noise_factor=0.3,    # 30% de ruido
        learning_rate=1e-3
    )
    
    # ========================================================================
    # PASO 4: Entrenar modelo completo
    # ========================================================================
    net.train(
        train_drug=traindata, 
        train_rna=train_rnadata,
        val_drug=testdata, 
        val_rna=test_rnadata
    )
    
    # ========================================================================
    # PASO 5: Guardar modelo
    # ========================================================================
    net.save_model()
    print("=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
