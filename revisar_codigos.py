
from pathlib import Path
import pandas as pd

archivo = Path("datos/RNT_Mompox_preparado.csv")

df = pd.read_csv(
    archivo,
    encoding="utf-8-sig",
    dtype={"codigo_rnt": "string"}
)

# Quitar espacios externos, sin cambiar los códigos
df["codigo_rnt"] = df["codigo_rnt"].str.strip()

print("\n=== 1. CÓDIGOS MÁS REPETIDOS ===")
print(df["codigo_rnt"].value_counts().head(15))

print("\n=== 2. EJEMPLOS DE REGISTROS REPETIDOS ===")
codigos_repetidos = (
    df["codigo_rnt"].value_counts()
    .loc[lambda x: x > 1]
    .head(5)
    .index
)

columnas = [
    "codigo_rnt",
    "razon_social_establecimiento",
    "categoria",
    "numero_de_habitaciones",
    "numero_de_camas",
    "numero_de_empleados",
    "año",
]

print(
    df[df["codigo_rnt"].isin(codigos_repetidos)][columnas]
    .sort_values(["codigo_rnt", "año"])
    .to_string(index=False)
)

print("\n=== 3. DUPLICADOS POR CÓDIGO Y AÑO ===")
print(
    "Filas repetidas en la combinación código-año:",
    df.duplicated(["codigo_rnt", "año"]).sum()
)

print("\n=== 4. AÑOS POR CÓDIGO ===")
años_por_codigo = df.groupby("codigo_rnt")["año"].nunique()

print("Códigos distintos:", df["codigo_rnt"].nunique())
print("Códigos presentes en varios años:", (años_por_codigo > 1).sum())
print("Máximo de años por código:", años_por_codigo.max())

print("\n=== 5. IDENTIFICADORES Y NOMBRES ===")
print(
    df[["codigo_rnt", "razon_social_establecimiento"]]
    .head(20)
    .to_string(index=False)
)
