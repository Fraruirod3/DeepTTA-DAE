# -*- coding: utf-8 -*-
"""
graficas_experimentos.py
========================
Lee las predicciones y métricas de todos los experimentos y genera:
  1. Barras comparativas de mejores métricas (Pearson, MSE, Spearman, CI)
  2. Curvas de entrenamiento (Pearson y MSE por época) superpuestas
  3. Scatter plot Label vs Pred para cada experimento
  4. Scatter plots por tipo de cáncer (solo si TCGA_DESC disponible)

Uso:
    python graficas_experimentos.py

Autor: Francisco Javier Ruiz Rodríguez
TFM - Universidad de Sevilla, 2025
"""

# Evitar error de caché de fuentes en disco lleno
import os, tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Para servidores sin display
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index

# ── Detección de entorno ──
if os.path.exists('/mnt/frareuirod3'):
    PROJECT_ROOT = '/mnt/frareuirod3/DeepTTA-TFG'
elif os.path.exists('/content/drive'):
    PROJECT_ROOT = '/content/drive/MyDrive/TFM/DeepTTC-main'
else:
    PROJECT_ROOT = 'C:/Users/franc/Desktop/TFM - Actualizado'

EXP_DIR = os.path.join(PROJECT_ROOT, 'Experimentos')
OUT_DIR = os.path.join(PROJECT_ROOT, 'Graficas_Comparativas')
os.makedirs(OUT_DIR, exist_ok=True)

# Estilo global
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


# ==========================================================================
# UTILIDADES
# ==========================================================================
def parse_validation_table(filepath):
    """Lee valid_markdowntable.txt y devuelve un DataFrame con métricas por época."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|'):
                continue
            if 'epoch' not in line.lower() or '#' in line:
                continue
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) < 8:
                continue
            try:
                epoch = int(re.search(r'\d+', parts[0]).group())
                rows.append({
                    'epoch': epoch,
                    'MSE': float(parts[1]),
                    'RMSE': float(parts[2]),
                    'Pearson': float(parts[3]),
                    'Spearman': float(parts[5]),
                    'CI': float(parts[7]),
                    'LR': parts[8] if len(parts) > 8 else ''
                })
            except (ValueError, AttributeError):
                continue
    return pd.DataFrame(rows)


def short_name(exp_name):
    """Nombre corto legible para gráficas."""
    name = exp_name.replace('exp_', '')
    # Extraer piezas clave
    parts = name.split('_')
    model = parts[0]  # MLP o DAE
    split = parts[1] if len(parts) > 1 else ''

    if model == 'DAE':
        noise = [p for p in parts if p.startswith('noise')]
        noise_str = noise[0] if noise else ''
        sched = 'ReduceLR' if 'ReduceLROnPlateau' in name else 'NoSched'
        return f"DAE {split} {noise_str} {sched}"
    else:
        sched = 'ReduceLR' if 'ReduceLROnPlateau' in name else 'NoSched'
        return f"MLP {split} {sched}"


def load_all_experiments():
    """Carga config y métricas de todos los experimentos."""
    experiments = []
    for d in sorted(os.listdir(EXP_DIR)):
        exp_path = os.path.join(EXP_DIR, d)
        if not os.path.isdir(exp_path) or not d.startswith('exp_'):
            continue

        cfg_path = os.path.join(exp_path, 'config.json')
        val_path = os.path.join(exp_path, 'valid_markdowntable.txt')

        if not os.path.exists(cfg_path) or not os.path.exists(val_path):
            continue

        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        df_val = parse_validation_table(val_path)
        if df_val.empty:
            continue

        experiments.append({
            'name': d,
            'short': short_name(d),
            'config': cfg,
            'metrics': df_val,
            'path': exp_path,
            'best_pearson': df_val['Pearson'].max(),
            'best_mse': df_val['MSE'].min(),
            'best_spearman': df_val['Spearman'].max(),
            'best_ci': df_val['CI'].max(),
            'best_rmse': df_val['RMSE'].min(),
        })

    return experiments


# ==========================================================================
# GRÁFICA 1: BARRAS COMPARATIVAS
# ==========================================================================
def plot_comparison_bars(experiments):
    """Barras horizontales comparando el mejor Pearson/MSE de cada exp."""
    names = [e['short'] for e in experiments]
    metrics = {
        'Pearson': [e['best_pearson'] for e in experiments],
        'MSE': [e['best_mse'] for e in experiments],
        'Spearman': [e['best_spearman'] for e in experiments],
        'CI': [e['best_ci'] for e in experiments],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors_dae = '#2ecc71'
    colors_mlp = '#3498db'

    for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
        colors = [colors_dae if 'DAE' in n else colors_mlp for n in names]
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_title(f'Mejor {metric} por Experimento', fontweight='bold')

        # Anotar valores
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

        ax.invert_yaxis()

    # Leyenda
    from matplotlib.patches import Patch
    legend = [Patch(color=colors_dae, label='DAE'),
              Patch(color=colors_mlp, label='MLP')]
    axes[0, 0].legend(handles=legend, loc='lower right')

    plt.suptitle('Comparativa de Experimentos — Mejores Métricas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '01_comparativa_barras.png'))
    plt.close()
    print("✅ 01_comparativa_barras.png")


# ==========================================================================
# GRÁFICA 2: CURVAS DE ENTRENAMIENTO
# ==========================================================================
def plot_training_curves(experiments):
    """Pearson y MSE por época, todos los experimentos superpuestos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cmap = plt.cm.get_cmap('tab10', len(experiments))

    for i, exp in enumerate(experiments):
        df = exp['metrics']
        color = cmap(i)
        label = exp['short']
        ax1.plot(df['epoch'], df['Pearson'], color=color, label=label, alpha=0.8)
        ax2.plot(df['epoch'], df['MSE'], color=color, label=label, alpha=0.8)

    # Paper reference line
    ax1.axhline(y=0.941, color='red', linestyle='--', alpha=0.5, label='Paper (0.941)')

    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pearson')
    ax1.set_title('Correlación de Pearson por Época', fontweight='bold')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Época')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE por Época', fontweight='bold')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Curvas de Entrenamiento', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '02_curvas_entrenamiento.png'))
    plt.close()
    print("✅ 02_curvas_entrenamiento.png")


# ==========================================================================
# GRÁFICA 3: CURVAS SEPARADAS POR SPLIT
# ==========================================================================
def plot_curves_by_split(experiments):
    """Curvas separadas para ByCancer y Random95."""
    for split in ['ByCancer', 'Random95']:
        exps = [e for e in experiments if e['config']['data_split'] == split]
        if not exps:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        cmap = plt.cm.get_cmap('tab10', len(exps))

        for i, exp in enumerate(exps):
            df = exp['metrics']
            color = cmap(i)
            label = exp['short']
            ax1.plot(df['epoch'], df['Pearson'], color=color, label=label, alpha=0.8)
            ax2.plot(df['epoch'], df['MSE'], color=color, label=label, alpha=0.8)

        if split == 'Random95':
            ax1.axhline(y=0.941, color='red', linestyle='--', alpha=0.5, label='Paper (0.941)')

        ax1.set_xlabel('Época'); ax1.set_ylabel('Pearson')
        ax1.set_title(f'Pearson — {split}', fontweight='bold')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Época'); ax2.set_ylabel('MSE')
        ax2.set_title(f'MSE — {split}', fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Comparativa {split}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = f'03_curvas_{split}.png'
        plt.savefig(os.path.join(OUT_DIR, fname))
        plt.close()
        print(f"✅ {fname}")


# ==========================================================================
# GRÁFICA 4: SCATTER LABEL vs PRED
# ==========================================================================
def plot_scatter_predictions(experiments):
    """Scatter Label vs Pred para cada experimento que tenga predicciones.csv."""
    for exp in experiments:
        pred_path = os.path.join(exp['path'], 'predicciones.csv')
        if not os.path.exists(pred_path):
            continue

        df = pd.read_csv(pred_path)
        if 'Label' not in df.columns or 'Pred' not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(df['Label'], df['Pred'], s=3, alpha=0.3, color='#8e44ad')

        # Línea 1:1
        lims = [min(df['Label'].min(), df['Pred'].min()),
                max(df['Label'].max(), df['Pred'].max())]
        ax.plot(lims, lims, '--', color='gray', alpha=0.7, label='1:1')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', 'box')

        # Métricas
        mse = mean_squared_error(df['Label'], df['Pred'])
        pear = linregress(df['Label'], df['Pred']).rvalue
        sp, _ = spearmanr(df['Label'], df['Pred'])
        ci = concordance_index(df['Label'], df['Pred'])

        text = f"MSE: {mse:.4f}\nPearson: {pear:.4f}\nSpearman: {sp:.4f}\nCI: {ci:.4f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Valor Real (IC50)'); ax.set_ylabel('Predicción')
        ax.set_title(exp['short'], fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe = exp['name'].replace(' ', '_')
        fname = f'04_scatter_{safe}.png'
        plt.savefig(os.path.join(OUT_DIR, fname))
        plt.close()
        print(f"✅ {fname}")


# ==========================================================================
# GRÁFICA 5: TABLA RESUMEN
# ==========================================================================
def plot_summary_table(experiments):
    """Genera una tabla visual con las mejores métricas."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(experiments)*0.6 + 1)))
    ax.axis('off')

    headers = ['Experimento', 'Split', 'Modelo', 'Scheduler', 'MSE', 'RMSE', 'Pearson', 'Spearman', 'CI']
    cell_data = []
    for e in experiments:
        cfg = e['config']
        sched = cfg.get('scheduler', 'None') or 'None'
        cell_data.append([
            e['short'], cfg['data_split'], cfg['model_type'], sched,
            f"{e['best_mse']:.4f}", f"{e['best_rmse']:.4f}",
            f"{e['best_pearson']:.4f}", f"{e['best_spearman']:.4f}",
            f"{e['best_ci']:.4f}"
        ])

    # Color rows
    colors = []
    for e in experiments:
        if e['config']['model_type'] == 'DAE':
            colors.append(['#d5f5e3'] * len(headers))
        else:
            colors.append(['#d6eaf8'] * len(headers))

    table = ax.table(cellText=cell_data, colLabels=headers,
                     cellColours=colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Header color
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('Resumen de Experimentos — Mejores Métricas', fontsize=13,
              fontweight='bold', pad=20)
    plt.savefig(os.path.join(OUT_DIR, '05_tabla_resumen.png'))
    plt.close()
    print("✅ 05_tabla_resumen.png")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GENERANDO GRÁFICAS COMPARATIVAS")
    print(f"Salida: {OUT_DIR}")
    print("=" * 70)

    experiments = load_all_experiments()
    print(f"Cargados {len(experiments)} experimentos\n")

    for e in experiments:
        print(f"  {e['short']:40s} Pearson={e['best_pearson']:.4f}  MSE={e['best_mse']:.4f}")

    print()
    plot_comparison_bars(experiments)
    plot_training_curves(experiments)
    plot_curves_by_split(experiments)
    plot_scatter_predictions(experiments)
    plot_summary_table(experiments)

    print(f"\n{'='*70}")
    print(f"GRÁFICAS GUARDADAS EN: {OUT_DIR}")
    print(f"{'='*70}")
