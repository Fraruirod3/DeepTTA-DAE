import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Datos de los 9 experimentos (verificados contra las tablas del TFM)
# ============================================================
experimentos = [
    "DAE ByCancer noise0.1 ReduceLR",
    "DAE ByCancer noise0.3 ReduceLR",
    "DAE Random95 noise0.1 NoSched",
    "DAE Random95 noise0.1 ReduceLR",   # seed=1, el mejor
    "DAE Random95 noise0.1 ReduceLR (seed2)",
    "MLP ByCancer NoSched",
    "MLP ByCancer ReduceLR",
    "MLP Random95 NoSched",
    "MLP Random95 ReduceLR",
]

es_dae = [True, True, True, True, True, False, False, False, False]

pearson  = [0.9465, 0.9459, 0.9488, 0.9491, 0.9478, 0.9384, 0.9385, 0.9417, 0.9416]
mse      = [0.8333, 0.8433, 0.7839, 0.7808, 0.8040, 0.9554, 0.9561, 0.8893, 0.8969]
spearman = [0.9233, 0.9217, 0.9277, 0.9281, 0.9267, 0.9102, 0.9101, 0.9186, 0.9169]
ci       = [0.8856, 0.8844, 0.8892, 0.8894, 0.8880, 0.8759, 0.8760, 0.8822, 0.8812]

colores = ["#2ecc71" if dae else "#3498db" for dae in es_dae]


def grafica_barras(valores, titulo, nombre_archivo, xlim=None, es_mse=False):
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(experimentos))

    ax.barh(y_pos, valores, color=colores)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(experimentos)
    ax.invert_yaxis()  # el primero arriba
    ax.set_xlabel(titulo)
    ax.set_title(f"Mejor {titulo} por Experimento", fontsize=14, fontweight="bold")

    if xlim:
        ax.set_xlim(xlim)

    # Etiqueta con el valor exacto al final de cada barra
    for i, v in enumerate(valores):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01 if xlim else v + 0.01,
                 i, f"{v:.4f}", va="center", fontsize=9)

    # Leyenda manual
    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor="#2ecc71", label="DAE"),
               Patch(facecolor="#3498db", label="MLP")]
    ax.legend(handles=leyenda, loc="lower right")

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {nombre_archivo}")


# MSE: rango real 0.78–0.96 → recorto un poco, pero menos agresivo (tiene más dispersión)
grafica_barras(mse, "MSE", "comparativa_mse.png", xlim=(0.70, 1.00))

# Pearson: rango real 0.938–0.949 → recorte fuerte, aquí es donde más se necesita
grafica_barras(pearson, "Pearson", "comparativa_pearson.png", xlim=(0.92, 0.96))

# Spearman: rango real 0.910–0.928
grafica_barras(spearman, "Spearman", "comparativa_spearman.png", xlim=(0.90, 0.94))

# CI: rango real 0.876–0.889
grafica_barras(ci, "Concordance Index", "comparativa_ci.png", xlim=(0.87, 0.90))