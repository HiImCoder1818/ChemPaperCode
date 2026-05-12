import pandas as pd

df = pd.read_csv("argon.csv", dtype=str)

ions = {}

for _, row in df.iterrows():
    sp = row["sp_num"]
    wl = row["obs_wl_air(nm)"]

    if int(float(sp)) != 2:
        continue

    if pd.isna(sp) or pd.isna(wl):
        continue

    # clean annoying formatting like ="300.1"
    sp = sp.replace('"','').replace('=','')
    wl = wl.replace('"','').replace('=','')

    if wl == "":
        continue

    ion = f"Ar {int(float(sp))-1}+"   # sp_num → ion label

    if ion not in ions:
        ions[ion] = []

    ions[ion].append(float(wl))


for ion in ions:
    print(ion)
    print(sorted(ions[ion]))
    print()