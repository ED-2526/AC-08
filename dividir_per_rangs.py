import pandas as pd
import numpy as np
import os

INPUT_CSV = "Data_Train.csv"
PRICE_COL = "Price"
OUTPUT_DIR = "dataset_rangs/"
OUTPUT_PREFIX = "vols_rang_"

## Carreguem el CSV
df = pd.read_csv(INPUT_CSV)

## Comprobem que la columna de preus és numèrica
df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
## Eliminem files sense preu
df = df.dropna(subset=[PRICE_COL])

## Definim els rangs de preus
bins = [0, 5000, 7500, 10000, 12500, 15000, np.inf]
labels = ["0-5000", "5000-7500", "7500-10000", "10000-12500", "12500-15000", ">=15000"]

## Creem la carpeta si no existeix
os.makedirs(OUTPUT_DIR, exist_ok=True)

## Assignar rangs de preu
df["price_range"] = pd.cut(df[PRICE_COL], bins=bins, labels=labels, right=True, include_lowest=True)

## Guardem els grups per rang
for label in labels:
    g = df[df["price_range"] == label].copy()
    safe_label = label.replace(">=", "gte_").replace(" ", "").replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{safe_label}.csv")
    g.to_csv(out_path, index=False)
    print(f"Guardat: {out_path} (files: {len(g)})")

## Resum de preus per grup
summary = pd.DataFrame({
    "Grup": labels,
    "Files": [len(df[df["price_range"] == label]) for label in labels],
    "preu_minim": [df[df["price_range"] == label][PRICE_COL].min() for label in labels],
    "preu_maxim": [df[df["price_range"] == label][PRICE_COL].max() for label in labels],
})

print("\nResum de preus per grup:\n", summary.to_string(index=False))