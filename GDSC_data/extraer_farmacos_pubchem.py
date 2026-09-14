"""
extraer_farmacos_pubchem.py

Genera el listado de fármacos empleados en el TFM (Anexo B de la memoria),
cruzando el archivo de respuestas de dosis de GDSC2 con el archivo de
anotación de fármacos, para obtener el identificador PubChem CID de cada uno.

Entradas:
    - GDSC_data/GDSC2_fitted_dose_response_25Feb20.xlsx  (respuestas de dosis, 198 fármacos)
    - GDSC_data/Drug_listTue_Aug10_2021.csv              (anotación, incluye PubChem CID)

Salida:
    - GDSC_data/listado_farmacos_anexo_FINAL.csv  (154 fármacos con PubChem CID válido)
"""

import pandas as pd

RUTA_DOSIS = "GDSC_data/GDSC2_fitted_dose_response_25Feb20.xlsx"
RUTA_ANOTACION = "GDSC_data/Drug_listTue_Aug10_2021.csv"
RUTA_SALIDA = "GDSC_data/listado_farmacos_anexo_FINAL.csv"


def cargar_farmacos_gdsc2(ruta: str) -> pd.DataFrame:
    """Carga el archivo de dosis-respuesta y devuelve los fármacos únicos de GDSC2."""
    df = pd.read_excel(ruta)
    farmacos = df[["DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"]]
    farmacos = farmacos.drop_duplicates(subset=["DRUG_ID"])
    return farmacos.sort_values("DRUG_ID").reset_index(drop=True)


def cargar_anotacion_pubchem(ruta: str) -> pd.DataFrame:
    """Carga el archivo de anotación y devuelve drug_id + PubChem, sin duplicados."""
    df = pd.read_csv(ruta)
    anotacion = df[["drug_id", "PubCHEM"]].drop_duplicates(subset=["drug_id"], keep="first")
    return anotacion


def cruzar_y_filtrar(farmacos: pd.DataFrame, anotacion: pd.DataFrame) -> pd.DataFrame:
    """Cruza ambos archivos por drug_id y filtra solo los que tienen PubChem CID válido."""
    resultado = farmacos.merge(
        anotacion, left_on="DRUG_ID", right_on="drug_id", how="left"
    )

    resultado["PubCHEM"] = resultado["PubCHEM"].astype(str).str.strip()

    # Un PubChem CID válido: no vacío, no "-", no "nan", no "none"
    validos = resultado[
        resultado["PubCHEM"].notna()
        & (resultado["PubCHEM"] != "-")
        & (resultado["PubCHEM"] != "")
        & (~resultado["PubCHEM"].str.lower().isin(["nan", "none"]))
    ].copy()

    # Algunos fármacos tienen varios CID separados por coma (sinónimos químicos);
    # nos quedamos con el primero para que el listado sea unívoco
    validos["PubCHEM"] = validos["PubCHEM"].apply(lambda x: x.split(",")[0].strip())

    return validos[["DRUG_ID", "DRUG_NAME", "PubCHEM"]].sort_values("DRUG_ID").reset_index(drop=True)


def main():
    farmacos = cargar_farmacos_gdsc2(RUTA_DOSIS)
    print(f"Fármacos únicos en GDSC2 (dose-response): {len(farmacos)}")

    anotacion = cargar_anotacion_pubchem(RUTA_ANOTACION)
    print(f"Fármacos únicos en el archivo de anotación: {len(anotacion)}")

    tabla_final = cruzar_y_filtrar(farmacos, anotacion)
    print(f"Fármacos con PubChem CID válido tras el cruce: {len(tabla_final)}")

    tabla_final.to_csv(RUTA_SALIDA, index=False)
    print(f"Guardado en: {RUTA_SALIDA}")

    print("\n--- Filas en formato LaTeX ---")
    for _, fila in tabla_final.iterrows():
        print(f"{fila['DRUG_ID']} & {fila['DRUG_NAME']} & {fila['PubCHEM']} \\\\")


if __name__ == "__main__":
    main()