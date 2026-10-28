

from pathlib import Path
import pandas as pd

archivo = Path("datos/RNT_Mompox_preparado.csv")

if not archivo.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {archivo}")

# Conservar el código RNT como texto para no alterarlo
df = pd.read_csv(
    archivo,
    encoding="utf-8-sig",
    dtype={"codigo_rnt": "string"}
)

print("\n=== DIAGNÓSTICO DE CALIDAD ===")
print(f"Registros: {len(df)}")
print(f"Columnas: {len(df.columns)}")
print(f"Filas completamente duplicadas: {df.duplicated().sum()}")
print(f"Códigos RNT duplicados: {df['codigo_rnt'].duplicated().sum()}")

print("\n=== VALORES ÚNICOS ===")
for columna in [
    "estado_rnt",
    "departamento",
    "municipio",
    "categoria",
    "sub_categoria",
    "ofrece_alojamiento",
]:
    print(f"\n{columna}:")
    print(df[columna].value_counts(dropna=False).head(15))

print("\n=== ESTADÍSTICAS NUMÉRICAS ===")
columnas = [
    "numero_de_habitaciones",
    "numero_de_camas",
    "numero_de_empleados",
    "año",
]

print(df[columnas].describe().round(2))

print("\n=== MUESTRA DE CÓDIGOS RNT ===")
print(df["codigo_rnt"].head(10).tolist())

print("\n=== DIAGNÓSTICO TERMINADO ===")
