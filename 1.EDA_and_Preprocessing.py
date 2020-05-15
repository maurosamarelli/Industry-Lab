#########################################################################################
#########################--- 0. IMPORT LIBRARIES & PACKAGES---###########################
#########################################################################################
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from Config_File import Dict_General
from Config_File import Dict_EDA_Prepro
from sklearn.impute import SimpleImputer
import warnings
warnings.simplefilter("ignore")

#########################################################################################
#######################--- 1. IMPORT DATA & SET-UP PARAMETERS ---########################
#########################################################################################
df = pd.read_csv(Path(Dict_General["path_data"]).joinpath("caso2_ALL_ESITI.csv"))

#########################################################################################
#######################--- 2. EXPLORATORY DATA ANALYSIS ---########################
#########################################################################################
#####--- 2.0 Print some dataset info
print("Dataset dimension: ", df.shape)
print("Dataset dtypes: ",df.dtypes)
print("Dataset head",df.head())
print("Dataset describe: ",df.describe())

#####--- 2.1 Create list with interested variables
#####--- 2.2 Plot numeric variables
sns.set(color_codes=True)
fig, axes = plt.subplots(figsize=(20,30), nrows=int(len(Dict_EDA_Prepro["interested_vars"])/2+0.5), ncols=2)
for i, column in enumerate(Dict_EDA_Prepro["interested_vars"]):
    sns.distplot(df[column], ax=axes[i//2, i%2])
plt.savefig(Path(Dict_General["path_data"]).joinpath(("Variables' distribution.png")))

#####--- 2.3  Overview about target relationship
# 2.3.0 Filter df and then print correlation value
df = df[df["n_esito"] == Dict_EDA_Prepro["n_esito_filter"]] # perchè teniamo solo queste oss.?s
print("Correlation: ",df.corr()["media_pressione_velocita_a_regime"]["media_portata_velocita_1"])

# 2.3.1 Plot correlaton graph
plt.scatter(df.media_pressione_velocita_a_regime, df.media_portata_velocita_1)
plt.title("speed=140rpm, T=40°C")
plt.xlabel("outlet pressure [bar]")
plt.ylabel("GP flow rate [L/h]")
plt.savefig(Path(Dict_General["path_data"]).joinpath("Correlation_pressione_velocità.png"))

#####--- 2.4 Look for missing values
# 2.4.0
print("Missing values per variable: ",df.isnull().sum())

# 2.4.1 Fill missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=" ")
imputer.fit(df)
df_tr = pd.DataFrame(imputer.transform(df), columns=df.columns)
print(sum(df.isnull().sum()))

#####--- 2.5 Remove univariate outliers
print("Dataframe dimension with outliers: ",len(df_tr))

# 2.5.0 Tranform column dtype
for var in Dict_EDA_Prepro["interested_vars"]:
    if var != "n_esito":
        df_tr[var] = df_tr[var].astype(float)
# 2.5.1 Remove outliers
for var in Dict_EDA_Prepro["interested_vars"]:
    if (var != "n_esito" and df_tr[var].std() != 0):
        df_tr["zscore"] = pd.Series(np.abs(zscore(df_tr[var])))
        df_tr.drop(list(df_tr.query("zscore > 3.5").index), inplace=True)
        df_tr.drop("zscore", 1, inplace=True)
        print(var, len(df_tr[var]))

print("Dataframe dimension without outliers: ",df_tr.shape)

#####--- 2.6 Remove some osservations based on scatterplot
# 2.6.0 Filter df_tr based on media_portata_velocita_1 value
df_tr = df_tr[(df_tr["media_portata_velocita_1"] >= Dict_EDA_Prepro["media_portata_velocita_1_filter_1"]) &
                     (df_tr["media_portata_velocita_1"] <= Dict_EDA_Prepro["media_portata_velocita_1_filter_2"])]

# 2.6.1 Print correlation
print("Correlation: ", df_tr.corr()["media_pressione_velocita_a_regime"]["media_portata_velocita_1"])

# 2.6.3 Craete scatterplot
plt.scatter(df_tr.media_pressione_velocita_a_regime, df_tr.media_portata_velocita_1)
plt.title("speed=140rpm, T=40°C")
plt.xlabel("outlet pressure [bar]")
plt.ylabel("GP flow rate [L/h]")
plt.savefig(Path(Dict_General["path_data"]).joinpath("Correlation_after_NaN.png"))

#####--- 3.0 Export df_tr
df_tr.to_csv(Path(Dict_General["path_data"]).joinpath("caso_2_after_eda_prepro.csv"))
print("Exported df_tr")
print("Finished all EDA and preprocessing")