import pandas as pd

data = pd.read_csv("crush_dataset.csv")
print(data["likes_you"].value_counts())