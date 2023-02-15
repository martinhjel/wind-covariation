import pandas as pd

df_nve = pd.read_csv("data/havvind_nve.csv")
df_nve.head()

df_nve.columns

print()
for r in df_nve.iterrows():
    # print(f"{r["NAVN"]}")
    pass

# Sort from north to south
df_nve["lat_centroid"] = df_nve["latlon_centroid"].apply(
    lambda x: tuple(map(float, [i.replace("(", "").replace(")", "") for i in x.split(", ")]))[1]
)
df_nve = df_nve.sort_values("lat_centroid", ascending=False)


df_latex = df_nve[["NAVN", "AREAL", "GJVINDPROD"]]
col_rename = {"NAVN": "Geographical area", "AREAL": "Sea area (km$^2$)", "GJVINDPROD": "Capacity factor"}
df_latex = df_latex.rename(columns=col_rename)
df_latex = df_latex.set_index("Geographical area")

df_mean = pd.read_csv("data/mean_wind.csv", index_col=0)
df_latex["Capacity factor"] = df_mean["0"]

df_latex["Capacity factor"] = df_latex["Capacity factor"].apply(lambda x: f"{x:.3f}")
df_latex["Sea area (km$^2$)"] = df_latex["Sea area (km$^2$)"].apply(lambda x: f"{x:.0f}")

df_latex["Min-av-max depth below surface (m)"] = df_nve.apply(
    lambda x: f"{-x['MAXDYBDE']:3.0f} - {-x['GJDYBDE']:3.0f} - {-x['MINDYBDE']:3.0f}", axis=1
).values

df_latex["Min-max distance  from shore (km)"] = (
    df_nve[["MINAVSTKYST", "MAXAVSTKYST"]]
    .apply(lambda x: f"{x['MINAVSTKYST']:3.0f} - {x['MAXAVSTKYST']:3.0f}", axis=1)
    .values
)

df_latex["Theoretical capacity [GW]"] = "0.0"

print(
    df_latex.style.to_latex(
        hrules=True,
        label="tab:overview-offshore-wind",
        # escape=False,
        column_format="|"+"|".join(["l" for _ in df_latex.columns])+"|",
        caption="Overview of all the geographical areas pointed out as offshore wind power areas by the Norwegian coast.",
    )
)
