import pandas as pd
from pathlib import Path


def main() -> None:
    """Prepare health cost tensor data from Excel source."""
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading data from {file_path.name}...")
    df = pd.read_excel(file_path, sheet_name="Data")

    # Validate required columns
    required_cols = [
        "Periode",
        "Canton_ISO2",
        "Groupe_de_couts",
        "Prestations_brutes_par_assure",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Handle period format
    if pd.api.types.is_datetime64_any_dtype(df["Periode"]):
        df["Periode"] = pd.PeriodIndex(df["Periode"], freq="Q").astype(str)
        print(f"  Converted periods to quarterly format")

    # Convert values to numeric
    df = df[pd.to_numeric(df["Prestations_brutes_par_assure"], errors="coerce").notna()]
    df["Prestations_brutes_par_assure"] = df["Prestations_brutes_par_assure"].astype(
        float
    )

    # Filter data - exclude aggregated categories
    df_model = df[
        ~df["Groupe_de_couts"].isin(
            [
                "Total",
                "Médicaments (médecin, pharmacie)",
            ]
        )
    ]

    print(f"\n📋 Data summary:")
    print(f"   Detail cost groups: {df_model['Groupe_de_couts'].nunique()}")
    print(f"   Cantons: {df_model['Canton_ISO2'].nunique()}")
    print(f"   Periods: {df_model['Periode'].nunique()}")
    print(
        f"   Period range: {df_model['Periode'].min()} to {df_model['Periode'].max()}"
    )

    # Check for duplicates
    duplicates = df_model.duplicated(
        subset=["Periode", "Canton_ISO2", "Groupe_de_couts"]
    )
    if duplicates.any():
        print(
            f"   ⚠️  Warning: {duplicates.sum()} duplicate rows found and will be handled"
        )

    # Pivot into tensor format
    tensor_df = df_model.pivot_table(
        index="Periode",
        columns=["Canton_ISO2", "Groupe_de_couts"],
        values="Prestations_brutes_par_assure",
        aggfunc="first",  # Take first value if duplicates exist
    )

    # Sort for consistency
    tensor_df = tensor_df.sort_index()
    tensor_df = tensor_df.sort_index(axis=1, level=[0, 1])

    # Flatten column MultiIndex to "Canton|Group" format
    tensor_df.columns = [f"{canton}|{group}" for canton, group in tensor_df.columns]

    # Data quality report
    print(f"\n📊 Output tensor:")
    print(f"   Shape: {tensor_df.shape[0]} periods × {tensor_df.shape[1]} series")
    n_missing = tensor_df.isna().sum().sum()
    pct_missing = 100 * tensor_df.isna().mean().mean()
    print(f"   Missing values: {n_missing} ({pct_missing:.1f}%)")

    # Save
    output_path = base_dir / "health_costs_tensor.csv"
    tensor_df.to_csv(output_path, index_label="Periode")
    print(f"\n✅ Saved to '{output_path.name}'")

    # Show sample
    print(f"\n📝 Sample data (first 3 periods, first 3 series):")
    print(tensor_df.iloc[:3, :3])


if __name__ == "__main__":
    main()
