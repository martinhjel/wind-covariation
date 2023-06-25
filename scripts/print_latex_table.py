import pandas as pd
from pathlib import Path
from _helpers import configure_logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("write_latex_table", area="norwegian", bias="bias_true")
    configure_logging(snakemake)

    result_table_file = Path(snakemake.input.alp_table)
    latex_table_file = Path(snakemake.output.latex_table)

    df_table = pd.read_csv(result_table_file, index_col=0)

    # Reorder from north to south
    # reorder_idx = df_table.iloc[:-2, :].index[::-1]
    # final_order_idx = reorder_idx.to_list() + df_table.iloc[-2:].index.to_list()
    # df_table = df_table.reindex(final_order_idx)

    # Round to two difigts
    df_table = df_table.round(2)

    # Renormalize so sum is 100
    diff = df_table.iloc[:-2, :].sum(axis=0) - 100
    first_with_value = (df_table.iloc[:-2, :] > 0.1).idxmax()

    for col, idx in first_with_value.items():
        df_table.loc[idx, col] -= diff[col]

    # Ensure that the file exist before writing latex table
    with open(latex_table_file, "w") as f:
        pass

    df_table.to_latex(
        buf=latex_table_file,
        index=True,
        column_format="|" + "|".join(["l" for _ in df_table.columns]) + "|",
        caption="Overview of how the model would weight the different wind farms given the different loss functions.",
        float_format="{:.2f}".format,
        label="tab:developments-by-loss-fn",
    )
