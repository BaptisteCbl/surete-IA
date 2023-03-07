import pandas as pd
import os


for filename in os.listdir("."):
    if filename.endswith(".csv"):

        processedname = os.path.splitext(filename)[0] + "_processed.csv"
        if not os.path.isfile("./" + processedname):
            print("Processing ", filename)
            df = pd.read_csv(filename, skiprows=[0])
            time = df[[" Epoch", "Time elapsed"]]
            time = time.groupby(" Epoch").tail(1).reset_index()
            df = df.groupby(" Epoch").mean()
            df["Time elapsed"] = time["Time elapsed"]

            df.to_csv(processedname, index=True)
