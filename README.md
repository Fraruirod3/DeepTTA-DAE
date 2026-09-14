# DeepTTA-DAE: Mejora del codificador transcriptómico en DeepTTA

Este repositorio contiene el código y experimentos del Trabajo Fin de Máster:

**"Data science en reposicionamiento de fármacos: deep learning e integración de 
información"**

Extiende un TFG previo centrado en DeepTTA [1], sustituyendo su codificador
transcriptómico (MLP) por un **Denoising AutoEncoder (DAE)** preentrenado de
forma no supervisada, validado mediante 9 experimentos y un caso de estudio
en glioblastoma.

---

## 🔍 Resumen

- **Modelo base:** DeepTTA (Transformer + MLP + codificación ESPF) [1]
- **Propuesta:** MLP → DAE preentrenado sobre los perfiles de expresión génica
- **Datos:** GDSC2 (103.492 pares célula-fármaco, 805 líneas celulares, 154 fármacos)
- **Resultados:**
  - ✅ Mejora consistente del DAE sobre el MLP (MSE, Pearson, Spearman, CI),
    bajo dos particiones (ByCancer 80/20 y Random 95/5), estadísticamente
    significativa (test de Wilcoxon, $p<0.001$)
  - ✅ Caso de estudio en glioblastoma validado frente a literatura biomédica
    y datos experimentales reales

Código y resultados organizados en `Experimentos/`, `Caso_estudio_GBM/`,
`Analisis_Cancer/` y `Graficas_Comparativas/`.

---

## 🧬 Origen del código

`model_helper.py`, `Step1_Cell_Stat.py`, `Step1_PubchemID2smile.py` y
`Step2_DataEncoding.py` proceden sin modificaciones sustanciales del
repositorio original de DeepTTA [1]. `dae_encoder.py`, `Step3_model_DAE.py`
y `experimento.py` son aportación de este TFM.

---

## 📎 Notas

Por su tamaño, estos archivos deben descargarse por separado:

1. Expresión génica de líneas celulares:
   https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip
2. Datos de glioblastoma (Neftel et al. [2]):
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131928
3. Pesos de los modelos entrenados (`.pt`, excluidos por tamaño): pueden
   regenerarse con `experimento.py` y el `config.json` de cada experimento
   en `Experimentos/`, o solicitarse al autor.

Este proyecto se presenta como Trabajo Fin de Máster en Ingeniería del
Software: Cloud, Datos y Gestión TI, Universidad de Sevilla.

[1] Jiang, L., Jiang, C., Yu, X., Fu, R., Jin, S., & Liu, X. (2022). DeepTTA:
a transformer-based model for predicting cancer drug response. *Briefings in
Bioinformatics*, 23(3), bbac100. https://doi.org/10.1093/bib/bbac100

[2] Neftel, C., Laffy, J., Filbin, M. G., et al. (2019). An Integrative Model
of Cellular States, Plasticity, and Genetics for Glioblastoma. *Cell*,
178(4), 835-849. https://doi.org/10.1016/j.cell.2019.06.024

---

Francisco Javier Ruiz Rodríguez
Máster en Ingeniería del Software: Cloud, Datos y Gestión TI
Universidad de Sevilla, 2026