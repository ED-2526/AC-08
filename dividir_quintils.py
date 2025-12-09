import pandas as pd
import os

INPUT_CSV = "Data_Train.csv"
OUTPUT_DIR = "dataset_quintils"
OUTPUT_PREFIX = "vols_part_"
PRICE_COL = "Price"

## Carregem el CSV
df = pd.read_csv(INPUT_CSV)

## Comprobem que la columna de preus és numèrica
df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")

## Eliminem files sense preu
df = df.dropna(subset=[PRICE_COL])

## Ordenem pel preu
df_sorted = df.sort_values(by=PRICE_COL).reset_index(drop=True)

## Dividim en 5 grups iguals en nombre de files
n = len(df_sorted)
k = 5
size = n // k
## Els primers 'remainder' grups tindran una fila extra
remainder = n % k

groups = []
start = 0
for i in range(k):
    ## Calculem l'índex final per a aquest grup
    extra = 1 if i < remainder else 0
    end = start + size + extra
    groups.append(df_sorted.iloc[start:end].copy())
    start = end

## Crear carpeta si no existeix
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, g in enumerate(groups, start=1):
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{i}.csv")
    g.to_csv(out_path, index=False)
    print(f"Guardat: {out_path} (files: {len(g)})")

## Resum de preus per grup
summary = pd.DataFrame({
    "Grup": [f"Q{i}" for i in range(1, k+1)],
    "Files": [len(g) for g in groups],
    "preu_minim": [g[PRICE_COL].min() for g in groups],
    "preu_maxim": [g[PRICE_COL].max() for g in groups],
})

print("\nResum de preus per grup:\n", summary.to_string(index=False))
