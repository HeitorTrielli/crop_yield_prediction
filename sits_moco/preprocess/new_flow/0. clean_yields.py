import pandas as pd

raw_yields = pd.read_excel(
    "files/raw_production_all_municipalities.xlsx", skiprows=3
).rename(columns={"Unnamed: 0": "municipality_code", "Unnamed: 1": "municipality_name"})

parana_municipalities = pd.read_csv("files/parana_municipalities.csv")

melted_yields = (
    raw_yields.melt(
        id_vars=["municipality_code", "municipality_name"],
        var_name="year",
        value_name="production",
    )
    .dropna()
    .assign(
        municipality_code=lambda x: x["municipality_code"].astype(int),
        production=lambda x: pd.to_numeric(x["production"], errors="coerce"),
    )
    .merge(parana_municipalities, on="municipality_code")
)

melted_yields.to_csv("files/municipality_production_with_codes.csv", index=False)
