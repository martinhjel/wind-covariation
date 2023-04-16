# pylint: skip-file

# --- Configurations and settings --- #
alpha_list = [0.5, 0.3, 0.1, 0.0]



# ---- Target rule  ----
rule all:
    input:
        "data/processed/augmented_lagrangian/latex_table.txt",
        "data/processed/wind_data.csv",
        "images/my_image.pdf"

# --- ETL rules ---

rule combine_wind_data:
    input:
        locations = "data/offshore_wind_locations.csv",
        nve_locations = "data/nve_offshore_wind_areas.csv"
    output:
        combined = "data/processed/wind_data.csv"
    shell:
        "python scripts/combine_wind_data.py --input-locations {input.locations} --input-nve-locations {input.nve_locations} --output-combined {output.combined}"

# --- Analysis rules ---
rule run_analysis:
    input:
        locations = "data/offshore_wind_locations.csv",
        nve_locations = "data/nve_offshore_wind_areas.csv",
        combined = "data/processed/wind_data.csv"
    shell:
        "python scripts/run_analysis.py --input-locations {input.locations} --input-nve-locations {input.nve_locations} --wind-data {input.combined}"

rule run_analysis_notebook:
    input:
        locations = "data/offshore_wind_locations.csv",
        nve_locations = "data/nve_offshore_wind_areas.csv",
        combined = "data/processed/wind_data.csv"
    output:
        "images/my_image.pdf"
    notebook:
        "notebooks/Analyze.py.ipynb"


# ---- Augmented Lagrangian rules ---
rule run_augmented_lagrangian:
    input:
        data_folder = "data/"
    output:
        model = "data/processed/augmented_lagrangian/alp_{alpha}/model.pkl"
    log:
        "logs/augmented_lagrangian_alp_{alpha}.log"
    params:
        alpha="{alpha}"
    shell:
        "python scripts/run_augmented_lagrangian.py --data-folder {input.data_folder} --models-output {output.model} --alpha {params.alpha}"


checkpoint combine_agumented_lagrangian_runs:
    input:
        model = expand("data/processed/augmented_lagrangian/alp_{alpha}/model.pkl", alpha=alpha_list)
    output:
        table = "data/processed/augmented_lagrangian/alp_table.pkl"
    log:
        "logs/combine_agumented_lagrangian_runs.log"
    params:
        alpha_list=alpha_list
    shell:
        "python scripts/combine_alpha_runs.py --input-model {input.model} --alpha {params.alpha_list} --output-table {output.table}"

rule write_latex_table:
    input:
        "data/processed/augmented_lagrangian/alp_table.pkl"
    output:
        "data/processed/augmented_lagrangian/latex_table.txt"
    log:
        "logs/write_latex_table.log"
    shell:
        "python scripts/print_latex_table.py --input {input} --output {output}"
