import pandas as pd
with open("./newlog.txt", "a") as f:
    f.writelines(f"{(pd.Timestamp.now().ceil(freq='H'))} : \n")
