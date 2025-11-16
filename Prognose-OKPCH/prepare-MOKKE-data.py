import pandas as pd
from pathlib import Path

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"

    df = pd.read_excel(file_path, sheet_name="Data")

    # Convert 'Prestations_brutes_par_assure' to numeric (handle 'na')
    df = df[pd.to_numeric(df["Prestations_brutes_par_assure"], errors="coerce").notna()]
    df["Prestations_brutes_par_assure"] = df["Prestations_brutes_par_assure"].astype(float)

    # Compute totals directly from the sum of all detailed Groupe_de_couts
    detail_rows = df[df["Groupe_de_couts"] != "Total"]
    totals = (
        detail_rows.groupby(["Periode", "Canton_ISO2"], as_index=False)[
            "Prestations_brutes_par_assure"
        ]
        .sum()
        .assign(Groupe_de_couts="Total")
    )

    df_full = pd.concat([detail_rows, totals], ignore_index=True)
    # Pivot into a 3D tensor stored in a flat CSV: Periode index, columns are
    # canton|groupe_de_couts pairs.
    tensor_df = df_full.pivot_table(
        index="Periode",
        columns=["Canton_ISO2", "Groupe_de_couts"],
        values="Prestations_brutes_par_assure",
    )
    tensor_df = tensor_df.sort_index()
    tensor_df = tensor_df.sort_index(axis=1, level=[0, 1])
    tensor_df.columns = [f"{canton}|{group}" for canton, group in tensor_df.columns]
    output_path = base_dir / "health_costs_tensor.csv"
    tensor_df.to_csv(output_path, index_label="Periode")
    print(f"Tensor saved to '{output_path.name}' with {tensor_df.shape[1]} columns")


if __name__ == "__main__":
    main()