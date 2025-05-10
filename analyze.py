"""
Load metrics .ccsv from runs, merge into a dataframe, compute statistics, generate comparative plots automatically. 
Results Collector/aggregator

TODO: Add more methods here to look at specific statistics or plots. 
"""
import pandas as pd
import glob
def load_all_results(pattern="runs/*/metrics.csv"):
    dfs = []
    for path in glob.glob(pattern):
        df = pd.read_csv(path)
        df["run"] = path.split("/")[1]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)