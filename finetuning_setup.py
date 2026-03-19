import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("DDI/ddi_metadata.csv", index_col=0)

# convert boolean to int
df["malignant"] = df["malignant"].astype(int)

# create skin tone groups
def tone_group(x):
    if x == 12:
        return "12"
    elif x == 34:
        return "34"
    elif x == 56:
        return "56"

df["tone_group"] = df["skin_tone"].apply(tone_group)

# create diagnosis label
df["diagnosis"] = df["malignant"].map({0: "benign", 1: "malignant"})

# create stratify column
df["stratify"] = df["tone_group"] + "_" + df["diagnosis"]

# check distribution
print(df["stratify"].value_counts())

# split 60/20/20
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["stratify"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["stratify"],
    random_state=42
)

train_df.to_csv("train.csv")
val_df.to_csv("val.csv")
test_df.to_csv("test.csv")

print("Split complete")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))