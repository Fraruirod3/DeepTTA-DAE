# -*- coding: utf-8 -*-
"""
DAE Encoder para DeepTTA-TFM
============================

Este módulo implementa un Denoising Autoencoder (DAE) que reemplaza al MLP simple
para la codificación de perfiles de expresión génica.

¿Qué es un DAE?
---------------
Un Autoencoder es una red neuronal que aprende a "comprimir" datos de alta dimensión
(17,737 genes) en un espacio de menor dimensión (256 dimensiones = "bottleneck") 
y luego "reconstruirlos".

La versión "Denoising" añade ruido a los datos de entrada durante el entrenamiento,
lo que obliga al modelo a aprender representaciones más robustas y generalizables.

Flujo del DAE:
    Input (17737 genes) 
        → Encoder (varias capas) 
            → Bottleneck (256 dim) 
                → Decoder (varias capas) 
                    → Output reconstruido (17737 genes)

¿Por qué es mejor que el MLP simple?
------------------------------------
1. PREENTRENAMIENTO: El DAE primero aprende la estructura de los datos de expresión
   génica (correlaciones entre genes, patrones de co-expresión) SIN saber nada de IC50.
   
2. ROBUSTEZ: Al entrenar con ruido, el modelo no memoriza datos individuales,
   sino que aprende patrones generales.

3. MEJOR GENERALIZACIÓN: Las representaciones aprendidas son más útiles para
   tareas downstream (predecir IC50).

Autor: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os

# ============================================================================
# CLASE PRINCIPAL: DAEEncoder
# ============================================================================

class DAEEncoder(nn.Module):
    """
    Denoising Autoencoder para codificar perfiles de expresión génica.
    
    Esta clase reemplaza al MLP simple de DeepTTA. La diferencia clave es que:
    - El MLP solo tiene forward pass (input → output)
    - El DAE tiene encoder + decoder y se preentrena antes del entrenamiento principal
    
    Parámetros:
    -----------
    input_dim : int
        Número de genes en el perfil de expresión (por defecto 17737, igual que GDSC)
    
    latent_dim : int
        Dimensión del espacio latente / bottleneck (por defecto 256, igual que el MLP original)
    
    hidden_dims : list of int
        Dimensiones de las capas ocultas del encoder (por defecto [1024, 512])
        El decoder usa las mismas dimensiones en orden inverso
    
    dropout : float
        Tasa de dropout para regularización (por defecto 0.3)
    """
    
    def __init__(self, 
                 input_dim=17737,      # Número de genes (igual que el MLP original)
                 latent_dim=256,        # Dimensión de salida (igual que hidden_dim_gene del MLP)
                 hidden_dims=[1024, 512],  # Capas ocultas del encoder
                 dropout=0.3):          # Regularización
        
        super(DAEEncoder, self).__init__()
        
        # Guardamos parámetros para uso posterior
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self._is_pretrained = False  # Flag para saber si ya se preentrenó
        
        # =====================================================================
        # CONSTRUCCIÓN DEL ENCODER
        # =====================================================================
        # El encoder reduce la dimensionalidad progresivamente:
        # input_dim → hidden_dims[0] → hidden_dims[1] → ... → latent_dim
        
        encoder_layers = []
        dims = [input_dim] + hidden_dims  # [17737, 1024, 512]
        
        for i in range(len(hidden_dims)):
            encoder_layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),      # Capa lineal (fully connected)
                nn.BatchNorm1d(dims[i+1]),          # Normalización por batch (estabiliza entrenamiento)
                nn.ReLU(),                          # Activación no lineal
                nn.Dropout(dropout)                 # Regularización (previene overfitting)
            ))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Capa bottleneck: última capa del encoder que produce el vector latente
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)  # 512 → 256
        
        # =====================================================================
        # CONSTRUCCIÓN DEL DECODER
        # =====================================================================
        # El decoder reconstruye los datos originales desde el espacio latente.
        # Usa las mismas dimensiones que el encoder, pero en orden inverso:
        # latent_dim → hidden_dims[-1] → hidden_dims[-2] → ... → input_dim
        
        # Primera capa del decoder: expande desde latent_dim
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])  # 256 → 512
        
        # Capas intermedias del decoder
        decoder_layers = []
        reversed_dims = hidden_dims[::-1] + [input_dim]  # [512, 1024, 17737]
        
        for i in range(len(reversed_dims) - 1):
            if i == len(reversed_dims) - 2:
                # Última capa: sin BatchNorm ni Dropout
                decoder_layers.append(nn.Sequential(
                    nn.Linear(reversed_dims[i], reversed_dims[i+1]),
                    nn.Sigmoid()  # Salida entre 0 y 1 (datos normalizados)
                ))
            else:
                decoder_layers.append(nn.Sequential(
                    nn.Linear(reversed_dims[i], reversed_dims[i+1]),
                    nn.BatchNorm1d(reversed_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    # =========================================================================
    # MÉTODO ENCODE: Lo que usará DeepTTA
    # =========================================================================
    def encode(self, x):
        """
        Codifica los datos de expresión génica en el espacio latente.
        
        Este es el método principal que usará DeepTTA después del preentrenamiento.
        Es equivalente al forward() del MLP original.
        
        Args:
            x: Tensor de expresión génica [batch_size, 17737]
        
        Returns:
            Tensor del espacio latente [batch_size, 256]
        """
        h = self.encoder(x)           # Pasa por todas las capas del encoder
        z = self.bottleneck(h)         # Proyecta al espacio latente
        return z
    
    # =========================================================================
    # MÉTODO DECODE: Solo se usa durante el preentrenamiento
    # =========================================================================
    def decode(self, z):
        """
        Reconstruye los datos originales desde el espacio latente.
        
        Solo se usa durante el PREENTRENAMIENTO para calcular el error
        de reconstrucción. No se usa durante la predicción de IC50.
        
        Args:
            z: Tensor del espacio latente [batch_size, 256]
        
        Returns:
            Tensor reconstruido [batch_size, 17737]
        """
        h = self.decoder_input(z)      # Expande desde latent_dim
        x_recon = self.decoder(h)      # Pasa por todas las capas del decoder
        return x_recon
    
    # =========================================================================
    # MÉTODO FORWARD: Para usar como reemplazo del MLP
    # =========================================================================
    def forward(self, v):
        """
        Forward pass para usar como reemplazo directo del MLP.
        
        Este método tiene la MISMA INTERFAZ que el MLP original:
        - Input: expresión génica
        - Output: vector latente de 256 dimensiones
        
        Args:
            v: Tensor de expresión génica [batch_size, 17737]
        
        Returns:
            Tensor del espacio latente [batch_size, 256]
        """
        v = v.float()
        return self.encode(v)
    
    # =========================================================================
    # MÉTODO PRETRAIN: Fase de preentrenamiento del DAE
    # =========================================================================
    def pretrain(self, 
                 data_loader, 
                 epochs=100, 
                 noise_factor=0.3,
                 learning_rate=1e-3,
                 device='cpu',
                 verbose=True):
        """
        Preentrena el DAE usando reconstrucción con ruido (Denoising).
        
        ¿Qué hace el preentrenamiento?
        -------------------------------
        1. Toma datos de expresión génica
        2. Añade ruido gaussiano (noise_factor controla cantidad)
        3. Intenta reconstruir los datos ORIGINALES (sin ruido)
        4. Minimiza el error de reconstrucción (MSE)
        
        Esto obliga al modelo a aprender representaciones robustas
        que capturan la estructura real de los datos, no el ruido.
        
        Args:
            data_loader: DataLoader con datos de expresión génica
            epochs: Número de épocas de preentrenamiento
            noise_factor: Cantidad de ruido gaussiano (0.3 = 30% del rango de datos)
            learning_rate: Tasa de aprendizaje
            device: 'cpu' o 'cuda'
            verbose: Si True, imprime progreso
        
        Returns:
            loss_history: Lista con pérdidas por época
        """
        
        self.to(device)
        self.train()  # Modo entrenamiento
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()  # Error cuadrático medio
        
        loss_history = []
        
        if verbose:
            print("=" * 60)
            print("PREENTRENAMIENTO DEL DAE")
            print("=" * 60)
            print(f"  Épocas: {epochs}")
            print(f"  Factor de ruido: {noise_factor}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Device: {device}")
            print("=" * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in data_loader:
                # Los datos pueden venir como (features,) o (features, labels)
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                
                x = x.float().to(device)
                
                # ============================================================
                # PASO CLAVE: AÑADIR RUIDO
                # ============================================================
                # El ruido gaussiano con media 0 y desviación estándar = noise_factor
                # hace que el modelo no pueda simplemente "copiar" la entrada
                x_noisy = x + noise_factor * torch.randn_like(x)
                
                # Asegurar que los valores estén en rango válido [0, 1]
                x_noisy = torch.clamp(x_noisy, 0, 1)
                
                # Forward pass: encode + decode
                z = self.encode(x_noisy)       # Codificar datos con ruido
                x_reconstructed = self.decode(z)  # Reconstruir
                
                # Calcular pérdida vs datos ORIGINALES (sin ruido)
                loss = loss_function(x_reconstructed, x)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Época {epoch+1}/{epochs} - Pérdida: {avg_loss:.6f}")
        
        self._is_pretrained = True
        
        if verbose:
            print("=" * 60)
            print("PREENTRENAMIENTO COMPLETADO")
            print(f"  Pérdida final: {loss_history[-1]:.6f}")
            print("=" * 60)
        
        return loss_history
    
    # =========================================================================
    # MÉTODOS DE GUARDADO Y CARGA
    # =========================================================================
    def save_pretrained(self, path):
        """
        Guarda los pesos del DAE preentrenado.
        
        Args:
            path: Ruta del archivo (ej: 'models/dae_pretrained.pt')
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'is_pretrained': self._is_pretrained
        }, path)
        print(f"DAE guardado en: {path}")
    
    def load_pretrained(self, path, device='cpu'):
        """
        Carga los pesos de un DAE preentrenado.
        
        Args:
            path: Ruta del archivo
            device: Device donde cargar el modelo
        
        Returns:
            self (para encadenamiento)
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self._is_pretrained = checkpoint.get('is_pretrained', True)
        print(f"DAE cargado desde: {path}")
        return self
    
    @property
    def is_pretrained(self):
        """Indica si el modelo ha sido preentrenado."""
        return self._is_pretrained


# ============================================================================
# CÓDIGO DE PRUEBA
# ============================================================================
if __name__ == '__main__':
    """
    Código de prueba para verificar que el DAE funciona correctamente.
    Puedes ejecutar este archivo directamente para probar:
    
        python dae_encoder.py
    """
    
    print("=" * 60)
    print("PRUEBA DEL DAE ENCODER")
    print("=" * 60)
    
    # Crear instancia del DAE
    dae = DAEEncoder(
        input_dim=17737,
        latent_dim=256,
        hidden_dims=[1024, 512],
        dropout=0.3
    )
    
    print(f"\n1. DAE creado con:")
    print(f"   - Input dim: {dae.input_dim}")
    print(f"   - Latent dim: {dae.latent_dim}")
    print(f"   - Hidden dims: {dae.hidden_dims}")
    
    # Crear datos de prueba aleatorios
    batch_size = 32
    x_test = torch.rand(batch_size, 17737)
    
    print(f"\n2. Datos de prueba: shape = {x_test.shape}")
    
    # Probar encode
    z = dae.encode(x_test)
    print(f"\n3. Encode: {x_test.shape} → {z.shape}")
    
    # Probar decode
    x_recon = dae.decode(z)
    print(f"   Decode: {z.shape} → {x_recon.shape}")
    
    # Probar forward (interfaz MLP)
    output = dae(x_test)
    print(f"\n4. Forward (interfaz MLP): {x_test.shape} → {output.shape}")
    
    # Verificar que output tiene las dimensiones correctas
    assert output.shape == (batch_size, 256), "¡Error en dimensiones de salida!"
    
    print("\n" + "=" * 60)
    print("✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
    print("=" * 60)
