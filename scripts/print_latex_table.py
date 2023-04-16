import pandas as pd
from pathlib import Path
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="")
parser.add_argument("--output", required=True, help="")
args = parser.parse_args()

result_table_file = Path(args.input)
latex_table_file = Path(args.output)

df_table = pd.read_csv(result_table_file, index_col=0)

# Ensure that the file exist before writing latex table
with open(latex_table_file, "w") as f:
    pass

df_table.to_latex(
    buf=latex_table_file,
    index=True,
    column_format="|" + "|".join(["l" for _ in df_table.columns]) + "|",
    caption="Overview of how the model would weight the different wind farms given the different loss functions.",
    float_format="{:.3f}".format,
    label="tab:developments-by-loss-fn",
)
