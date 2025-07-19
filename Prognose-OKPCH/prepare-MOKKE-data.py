import pandas as pd

# Load Excel file
file_path = "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"
df = pd.read_excel(file_path, sheet_name="Data")

# Convert 'Prestations_brutes_par_assure' to numeric (handle 'na')
df = df[pd.to_numeric(df['Prestations_brutes_par_assure'], errors='coerce').notna()]
df['Prestations_brutes_par_assure'] = df['Prestations_brutes_par_assure'].astype(float)

# Aggregate to get total cost per insured per canton per quarter
df_grouped = df.groupby(['Periode', 'Canton_ISO2'])['Prestations_brutes_par_assure'].sum().reset_index()

# Pivot to get time series matrix: rows = quarters, columns = cantons
matrix_df = df_grouped.pivot(index='Periode', columns='Canton_ISO2', values='Prestations_brutes_par_assure')

# Sort by time
matrix_df = matrix_df.sort_index()

# Save to CSV for modeling or forecasting
matrix_df.to_csv("health_costs_matrix.csv")

print("Matrix saved to 'health_costs_matrix.csv'")
