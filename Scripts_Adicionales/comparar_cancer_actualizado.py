"""
comparar_cancer_actualizado.py
================================
Compara el rendimiento del DAE frente al MLP, desagregado por tipo de
cáncer, para cada estrategia de partición (ByCancer y Random95),
usando los 9 experimentos oficiales del TFM. Incluye un test de
Wilcoxon para evaluar la significancia estadística de la mejora.

Requiere que cada carpeta de Experimentos/exp_... contenga predicciones.csv
(generado por predecir_experimentos.py).

Autor: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2026
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, wilcoxon
from lifelines.utils import concordance_index

import matplotlib
matplotlib.use('Agg')  # añadir junto a los imports del principio, evita problemas sin pantalla
import matplotlib.pyplot as plt

# ── Detección de entorno (mismo patrón que el resto de scripts del TFM) ──
if os.path.exists('/mnt/frareuirod3'):
    PROJECT_ROOT = '/mnt/frareuirod3/DeepTTA-TFG'
else:
    PROJECT_ROOT = 'E:/TFM - Actualizado'

EXP_DIR = os.path.join(PROJECT_ROOT, 'Experimentos')
OUT_DIR = os.path.join(PROJECT_ROOT, 'Analisis_Cancer')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pares (MLP, DAE) a comparar, uno por partición ──
COMPARACIONES = {
    'ByCancer': {
        'MLP': 'exp_MLP_ByCancer_NoScheduler_20260412_155057',
        'DAE': 'exp_DAE_ByCancer_pretrain100_noise0.1_drop0.3_ReduceLROnPlateau_20260412_211519',
    },
    'Random95': {
        'MLP': 'exp_MLP_Random95_NoScheduler_20260510_091725',
        'DAE': 'exp_DAE_Random95_pretrain100_noise0.1_drop0.3_ReduceLROnPlateau_20260509_205129',
    },
}


def calc_metrics(df):
    mse = mean_squared_error(df['Label'], df['Pred'])
    rmse = np.sqrt(mse)
    pearson = pearsonr(df['Label'], df['Pred'])[0]
    spearman = spearmanr(df['Label'], df['Pred'])[0]
    ci = concordance_index(df['Label'], df['Pred'])
    return {'MSE': mse, 'RMSE': rmse, 'Pearson': pearson, 'Spearman': spearman, 'CI': ci}


def cargar_predicciones(exp_name):
    path = os.path.join(EXP_DIR, exp_name, 'predicciones.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. ¿Ejecutaste predecir_experimentos.py?")
    df = pd.read_csv(path)
    if 'TCGA_DESC' not in df.columns:
        raise ValueError(f"{exp_name}: predicciones.csv no tiene columna TCGA_DESC")
    return df


def comparar_split(split_name, exp_mlp, exp_dae):
    print("=" * 80)
    print(f"COMPARACIÓN POR TIPO DE CÁNCER — {split_name}")
    print(f"  MLP: {exp_mlp}")
    print(f"  DAE: {exp_dae}")
    print("=" * 80)

    df_mlp = cargar_predicciones(exp_mlp)
    df_dae = cargar_predicciones(exp_dae)

    # Filtrar valores vacíos/NaN antes de construir el conjunto de tipos de cáncer
    tipos_mlp = set(df_mlp['TCGA_DESC'].dropna().unique())
    tipos_dae = set(df_dae['TCGA_DESC'].dropna().unique())
    tipos_cancer = sorted(tipos_mlp & tipos_dae, key=str)

    resultados = []
    for cancer in tipos_cancer:
        mlp_grupo = df_mlp[df_mlp['TCGA_DESC'] == cancer]
        dae_grupo = df_dae[df_dae['TCGA_DESC'] == cancer]

        # Se necesita un mínimo de muestras para que Pearson/Spearman/CI sean fiables
        if len(mlp_grupo) < 5 or len(dae_grupo) < 5:
            continue

        m_mlp = calc_metrics(mlp_grupo)
        m_dae = calc_metrics(dae_grupo)

        resultados.append({
            'Cancer': cancer,
            'N_MLP': len(mlp_grupo),
            'N_DAE': len(dae_grupo),
            'MLP_MSE': m_mlp['MSE'], 'DAE_MSE': m_dae['MSE'],
            'Diff_MSE': m_mlp['MSE'] - m_dae['MSE'],  # positivo = DAE mejor
            'MLP_Pearson': m_mlp['Pearson'], 'DAE_Pearson': m_dae['Pearson'],
            'Diff_Pearson': m_dae['Pearson'] - m_mlp['Pearson'],  # positivo = DAE mejor
        })

    df_resultados = pd.DataFrame(resultados).sort_values('Diff_MSE', ascending=False)

    dae_gana = (df_resultados['Diff_MSE'] > 0).sum()
    mlp_gana = (df_resultados['Diff_MSE'] < 0).sum()

    print(f"\n{'Cáncer':<15} {'N':>5} {'MLP_MSE':>8} {'DAE_MSE':>8} {'ΔMSE':>8} {'MLP_r':>7} {'DAE_r':>7} {'Δr':>6}")
    print("-" * 75)
    for _, row in df_resultados.iterrows():
        marca = "✅" if row['Diff_MSE'] > 0 else "❌"
        print(f"{row['Cancer']:<15} {row['N_MLP']:>5} {row['MLP_MSE']:>8.3f} {row['DAE_MSE']:>8.3f} "
              f"{row['Diff_MSE']:>+8.3f} {row['MLP_Pearson']:>7.3f} {row['DAE_Pearson']:>7.3f} "
              f"{row['Diff_Pearson']:>+6.3f} {marca}")

    print(f"\nDAE mejora el MSE en {dae_gana}/{len(df_resultados)} tipos de cáncer")
    print(f"MLP mejora el MSE en {mlp_gana}/{len(df_resultados)} tipos de cáncer")

    # ── Test de significancia estadística (Wilcoxon, datos emparejados) ──
    diff_mse = df_resultados['Diff_MSE'].values
    stat, p_value = wilcoxon(diff_mse, alternative='greater')

    print(f"\n--- Test de Wilcoxon (H1: DAE mejora MSE de forma sistemática) ---")
    print(f"Estadístico: {stat:.2f}")
    print(f"p-valor (unilateral): {p_value:.6f}")
    if p_value < 0.05:
        print("✅ Mejora estadísticamente significativa (p < 0.05)")
    else:
        print("⚠️ No significativa al nivel habitual (p ≥ 0.05)")

    out_path = os.path.join(OUT_DIR, f'comparacion_cancer_{split_name}.csv')
    df_resultados.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}\n")

    return df_resultados, p_value


def scatter_mse_por_cancer(df_resultados, split_name):
    fig, ax = plt.subplots(figsize=(7, 7))

    tamanos = df_resultados['N_MLP'] / 10
    colores = ['#2ecc71' if d > 0 else '#e74c3c' for d in df_resultados['Diff_MSE']]

    ax.scatter(df_resultados['MLP_MSE'], df_resultados['DAE_MSE'],
               s=tamanos, c=colores, alpha=0.6, edgecolors='black', linewidth=0.5)

    lims = [min(df_resultados['MLP_MSE'].min(), df_resultados['DAE_MSE'].min()) - 0.05,
            max(df_resultados['MLP_MSE'].max(), df_resultados['DAE_MSE'].max()) + 0.05]
    ax.plot(lims, lims, '--', color='gray', label='Igual rendimiento')

    ax.set_xlabel('MSE — MLP')
    ax.set_ylabel('MSE — DAE')
    ax.set_title(f'MSE por tipo de cáncer: MLP vs DAE — {split_name}\n(tamaño = nº de muestras)',
                 fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Aseguramos que la carpeta existe justo antes de guardar, y forzamos ruta absoluta normalizada
    os.makedirs(OUT_DIR, exist_ok=True)
    nombre_archivo = os.path.normpath(os.path.join(OUT_DIR, f'scatter_cancer_{split_name}.png'))
    
    print(f"Intentando guardar en: {nombre_archivo}")
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {nombre_archivo}")


if __name__ == '__main__':
    resultados_totales = {}
    for split_name, exps in COMPARACIONES.items():
        df_resultados, p_value = comparar_split(split_name, exps['MLP'], exps['DAE'])
        scatter_mse_por_cancer(df_resultados, split_name)
        resultados_totales[split_name] = {'df': df_resultados, 'p_value': p_value}

    print("=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    for split_name, datos in resultados_totales.items():
        print(f"{split_name}: p-valor = {datos['p_value']:.6f}")

    # ── Comprobar hipótesis: tamaño de muestra por tipo de cáncer ──
    df_resultados_bycancer = resultados_totales['ByCancer']['df']
    df_resultados_random95 = resultados_totales['Random95']['df']

    print("\nTamaño medio de test por tipo de cáncer:")
    print(f"  ByCancer: {df_resultados_bycancer['N_MLP'].mean():.0f}")
    print(f"  Random95: {df_resultados_random95['N_MLP'].mean():.0f}")