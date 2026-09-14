# DeepTTA - Adaptación y evaluación para reposicionamiento farmacológico

Este repositorio contiene el código y configuraciones utilizadas en el Trabajo de Fin de Grado:

**" Avances en Inteligencia Artificial en elcontexto del reposicionamiento de fármacos"**

El objetivo ha sido reproducir, analizar y evaluar el modelo DeepTTA [1], un sistema híbrido basado en Transformers y codificaciones químicas (ESPF), aplicado a la predicción de la sensibilidad celular a compuestos antitumorales.

---

## 🔍 Resumen del proyecto

- **Modelo estudiado:** DeepTTA (Transformer + MLP + codificación ESPF)
- **Datos utilizados:** Perfiles de expresión génica y estructuras moleculares (SMILES) del consorcio GDSC
- **Aportaciones principales:**
  - ✅ Reproducción funcional del modelo original en entorno local.
  - ⚖️ Diseño de particiones de datos balanceadas (`ByCancerEquilibrado`, `ByDrugEquilibrado`).
  - 📈 Visualización de resultados por tipo de cáncer y por fármaco.

- Todos los resultados obtenidos (predicciones, métricas y gráficas) se encuentran organizados en las carpetas 'predicciones' y 'graficas_predicciones' respectivamente .
---

## 📎 Notas

Debido al tamaño de los archivos, estos archivos deben descargarse por separado:

1. Archivo con las características génicas de las líneas celulares: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip
2. Modelos entrenados: [https://drive.google.com/drive/u/2/folders/1JWLLyCYW1wWNZBEsflt4l8aSdRoga72K](https://drive.google.com/file/d/1J5kJnznJ6JOSaDzutiu4XJI9GDGsbaco/view?usp=drive_link) cuyas informaciones están en la carpeta Registro_Modelos

Este proyecto es un estudio académico del modelo original DeepTTA [1] y se presenta como parte del Trabajo Fin de Grado (TFG) para la obtención del título en la carrera de Ingeniería de la Salud, mención Informática Clínica, por la Universidad de Sevilla. El código se ha reestructurado para facilitar la comprensión, experimentación y evaluación del modelo en un entorno reproducible, así como adaptado para una serie de experimentos que se presentan en la memoria del TFG presentado.
 
This project is an academic study of the original DeepTTA model [1]. It is presented as part of the bachelor thesis (TFG) in the degree of Ingeniería de la Salud - Informática Clínica in the  University of Seville. The code has been restructured to facilitate the understanding, experimentation and evaluation of the model in a 
reproducible environment, as well as adapted for a series of experiments that are presented in the memory of the bachelor thesis.
 
[1] Likun Jiang, Changzhi Jiang, Xinyu Yu, Rao Fu, Shuting Jin, Xiangrong Liu, DeepTTA: a transformer-based model for predicting cancer drug response, Briefings in Bioinformatics, Volume 23, Issue 3, May 2022, bbac100, https://doi.org/10.1093/bib/bbac100. Enlace: https://academic.oup.com/bib/article/23/3/bbac100/6554594 


Francisco Javier Ruiz Rodríguez
Grado en Ingeniería de la Salud – mención en Informática Clínica
Universidad de Sevilla, 2025
