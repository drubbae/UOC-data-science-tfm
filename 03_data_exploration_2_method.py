import pandas as pd
import matplotlib.pyplot as plt
print('package --> pandas --> version --> ' + pd.__version__)

debug = False

# ---------- STEP 01 ----------
# load parsed data
df = pd.read_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_11.csv')
# -----------------------------


# ---------- STEP 02 ----------
# null data
for col in df.columns:
    print(col, df[col].isnull().sum())


# ---------- STEP 03 ----------
# count/sum of rows by language
df.groupby(['language']).size().plot(kind='bar')
plt.show()