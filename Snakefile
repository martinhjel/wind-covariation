# pylint: skip-file
from glob import glob

# --- Configurations and settings --- #
alpha_list = [0.5, 0.3, 0.1, 0.0]

area_list = ["all", "norwegian"]
bias_list = ["bias_true", "bias_false"]


# ---- Target rule  ----
rule all:
    input:
        "data/processed/augmented_lagrangian/norwegian/bias_true/latex_table.tex",
        "data/processed/augmented_lagrangian/norwegian/bias_false/latex_table.tex",
        "data/processed/augmented_lagrangian/all/bias_false/latex_table.tex",
        "data/processed/wind_data.csv",
        "images/corr-distance.pdf",
        "images/shift-quantile0.9.pdf",
        "images/shift-quantile0.9999.pdf",
        "images/utsira-nord-std-wind-7D.pdf",
        "images/utsira-nord-std-wind-1H.pdf",
        "images/scatter-soerlige-nordsjoe-ii-nordmela.pdf",
        "images/scatter-soerlige-nordsjoe-ii-de-west.pdf",
        "images/corr-matrix-1day.pdf",
        "images/corr-matrix-1day-aggregated.pdf",
        "images/deviation-soerlige-nordsjoe-ii.pdf",
        [f"images/weights_training/{area}/{bias}/{alpha}_alpha.pdf" for alpha in alpha_list for area, bias in [("norwegian", "bias_true"), ("norwegian", "bias_false"), ("all", "bias_false")]]

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
rule run_analysis_legacy:
    input:
        locations = "data/offshore_wind_locations.csv",
        nve_locations = "data/nve_offshore_wind_areas.csv",
        combined = "data/processed/wind_data.csv"
    shell:
        "python scripts/run_analysis.py --input-locations {input.locations} --input-nve-locations {input.nve_locations} --wind-data {input.combined}"

rule run_analysis:
    input:
        locations = "data/offshore_wind_locations.csv",
        nve_locations = "data/nve_offshore_wind_areas.csv",
        combined = "data/processed/wind_data.csv",
        models = expand("data/processed/augmented_lagrangian/{area}/{bias}/alp_{alpha}/model.pkl", alpha=alpha_list, area="all", bias="bias_false")
    output:
        "images/corr-distance.pdf",
        "images/shift-quantile0.9.pdf",
        "images/shift-quantile0.9999.pdf",
        "images/utsira-nord-std-wind-7D.pdf",
        "images/utsira-nord-std-wind-1H.pdf",
        "images/scatter-soerlige-nordsjoe-ii-nordmela.pdf",
        "images/scatter-soerlige-nordsjoe-ii-de-west.pdf",
        "images/corr-matrix-1day.pdf",
        "images/corr-matrix-1day-aggregated.pdf",
        "images/deviation-soerlige-nordsjoe-ii.pdf"
    log:
        "logs/run_analysis.log"
    script:
        "scripts/run_analysis.py"

# ---- Augmented Lagrangian rules ---
rule run_augmented_lagrangian:
    input:
        data_folder = "data/"
    output:
        model = "data/processed/augmented_lagrangian/{area}/{bias}/alp_{alpha}/model.pkl"
        # model = "data/processed/augmented_lagrangian/norwegian/bias_true/alp_0.0/model.pkl"
    log:
        "logs/augmented_lagrangian_{area}_{bias}_alp_{alpha}.log"
    params:
        alpha="{alpha}",
        area="{area}",
        bias="{bias}"
    script:
        "scripts/run_augmented_lagrangian.py"

rule combine_agumented_lagrangian_runs:
    input:
        model = lambda wildcards: expand("data/processed/augmented_lagrangian/{area}/{bias}/alp_{alpha}/model.pkl",
                                    area=wildcards.area, bias=wildcards.bias, alpha=alpha_list)
    output:
        table = "data/processed/augmented_lagrangian/{area}/{bias}/alp_table.pkl"
    log:
        "logs/combine_agumented_lagrangian_runs_{area}_{bias}.log"
    params:
        alpha_list=alpha_list
    script:
        "scripts/combine_alpha_runs.py"

rule write_latex_table:
    input:
        alp_table = "data/processed/augmented_lagrangian/{area}/{bias}/alp_table.pkl"
    output:
        latex_table = "data/processed/augmented_lagrangian/{area}/{bias}/latex_table.tex"
    log:
        "logs/write_latex_table_{area}_{bias}.log"
    script:
        "scripts/print_latex_table.py"

rule plot_weights:
    input:
        model = "data/processed/augmented_lagrangian/{area}/{bias}/alp_{alpha}/model.pkl"
    output:
        "images/weights_training/{area}/{bias}/{alpha}_alpha.pdf"
    log:
        "logs/plot_weights_{area}_{bias}_{alpha}.log"
    script:
        "scripts/plot_weights.py"