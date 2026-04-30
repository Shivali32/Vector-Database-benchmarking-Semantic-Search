import duckdb
import pandas as pd

DB_PATH = "../../phase_2/benchmark.duckdb"
OUTPUT_CSV = "benchmark_results.csv"

conn = duckdb.connect(DB_PATH, read_only=True)

# Change table name if your table is named differently
table_name = "results"

df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

# Add index number as first column 
df.insert(0, "index", range(1, len(df) + 1))

# Export to CSV
df.to_csv(OUTPUT_CSV, index=False)

conn.close()

print(f"Exported {len(df)} rows to {OUTPUT_CSV}")