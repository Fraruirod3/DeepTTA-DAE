import pandas as pd
import numpy as np

# ============================================================
# 1. Cargar la matriz completa de GDSC2
# ============================================================
gdsc2 = pd.read_csv('Cell_line_RMA_proc_basalExp.txt', sep='\t', index_col=0)

if 'GENE_title' in gdsc2.columns:
    gdsc2 = gdsc2.drop(columns=['GENE_title'])

gene_order = gdsc2.index.tolist()
gene_means = gdsc2.mean(axis=1)

print(f"Genes en GDSC2 (orden original): {len(gene_order)}")
print(f"Columnas de expresión usadas: {gdsc2.shape[1]}")

# ============================================================
# 2. Cargar el perfil GBM — columna "Malignant cells"
# ============================================================
gbm = pd.read_csv('Glioma_GSE131928_Smartseq2_expression_Celltype_malignancy.txt',
                   sep='\t', index_col=0)
gbm_malignant = gbm['Malignant cells']

print(f"Genes en GBM: {len(gbm_malignant)}")

# ============================================================
# 3. Construir el vector final (17.737 dim, mismo orden que GDSC2)
# ============================================================
def valor_escalar(x):
    """Si x es una Series (por nombres duplicados), coge el primer valor."""
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

vector_final = []
genes_imputados = 0

for gen in gene_order:
    if gen in gbm_malignant.index:
        valor = valor_escalar(gbm_malignant.loc[gen])
    else:
        valor = valor_escalar(gene_means[gen])
        genes_imputados += 1
    vector_final.append(valor)

vector_final = np.array(vector_final, dtype=np.float64)

print(f"\nVector final construido: {vector_final.shape}")
print(f"Genes imputados con media GDSC2: {genes_imputados} "
      f"({genes_imputados/len(gene_order)*100:.2f}%)")
print(f"Genes tomados directamente del GBM: {len(gene_order) - genes_imputados}")

np.save('gbm_malignant_vector_imputado.npy', vector_final)
print("\nGuardado en: gbm_malignant_vector_imputado.npy")