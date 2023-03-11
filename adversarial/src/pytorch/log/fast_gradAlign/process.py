import pandas as pd
import os


for filename in os.listdir("."):

    if filename.endswith(".csv"):
        print(filename)
        processedname = os.path.splitext(filename)[0] + "_processed.csv"
        if not os.path.isfile("./" + processedname):
            print("Processing ", filename)
            df = pd.read_csv(filename, skiprows=[0])
            time = df[[" Epoch", "Time elapsed"]]
            time = time.groupby(" Epoch").tail(1).reset_index()
            df = df.groupby(" Epoch").mean()
            df["Time elapsed"] = time["Time elapsed"]
            for col in [
                "Train acc clean",
                "Train acc PGD",
                " Test acc clean",
                "Test acc FGSM",
                "Test acc PGD",
            ]:
                df[col] = df[col].apply(lambda x: x * 100)
            df.to_csv(processedname, index=True)
