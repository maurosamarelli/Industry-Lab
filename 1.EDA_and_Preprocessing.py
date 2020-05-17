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
from sklearn.linear_model import LinearRegression
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
print("\nDataset dtypes: ")
print(df.dtypes)
print("\nDataset head", df.head())
print("\nDataset describe: ", df.describe())

#####--- 2.1 Plot numeric interested variables
sns.set(color_codes=True)
fig, axes = plt.subplots(figsize=(20, 30), nrows=int(len(Dict_EDA_Prepro["interested_vars"])/2+0.5), ncols=2)
for i, column in enumerate(Dict_EDA_Prepro["interested_vars"]):
    sns.distplot(df[column], ax=axes[i//2, i % 2])
plt.savefig(Path(Dict_General["path_data"]).joinpath(("Variables' distribution.png")))
plt.close()

#####--- 2.2  Filter Dataset for a good evaluation
df = df.query("n_esito == " + Dict_EDA_Prepro["filter_n_esito"] +
              " and velicita_1 == " + Dict_EDA_Prepro["filter_velocita_1"] +
              " and velocita_a_regime == " + Dict_EDA_Prepro["filter_velocita_a_regime"] +
              " and Temperatura >= " + Dict_EDA_Prepro["filter_min_temperatura"] +
              " and Temperatura <= " + Dict_EDA_Prepro["filter_max_temperatura"] +
              " and " + Dict_EDA_Prepro["filter_positive_values"])

#####--- 2.3 Look for missing values
# 2.3.1
print("\nMissing values per variable: ")
print(df.isnull().sum())

# 2.3.2 Fill missing values
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=' ')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
print("Numero missing values:", sum(df.isnull().sum()))

#####--- 2.4 Remove univariate outliers
print("\nDataframe dimension with outliers: ", len(df))

# 2.4.1 Tranform column dtype
for var in Dict_EDA_Prepro["interested_vars"]:
    if var != "n_esito":
        df[var] = df[var].astype(float)

# 2.4.2 Remove outliers
for var in Dict_EDA_Prepro["interested_vars"]:
    if var != "n_esito":
        if df[var].std() != 0:
            df['zscore'] = pd.Series(np.abs(zscore(df[var])))
            idx = list(df.query("zscore > 3.5").index)
            print(var, len(idx))
            df.drop(idx, inplace=True)
            df.drop("zscore", 1, inplace=True)

print("Dataframe dimension without outliers: ", df.shape)

#####--- 2.5 Analyze Regime Phase @2300 rpm
print("\nCorrelazione grandezze fase di regime @2300rpm")
print("pressione - portata:\t\t", round(df.corr()['media_pressione_velocita_a_regime']['media_portata_velocita_a_regime'], 3))
print("coppia_zero - pressione:\t", round(df.corr()['media_coppia_zero']['media_pressione_velocita_a_regime'], 3))
print("coppia_zero - portata:\t\t", round(df.corr()['media_coppia_zero']['media_portata_velocita_a_regime'], 3))
print("coppia_finale - pressione:\t", round(df.corr()['media_coppia_finale']['media_pressione_velocita_a_regime'], 3))
print("coppia_finale - portata:\t", round(df.corr()['media_coppia_finale']['media_portata_velocita_a_regime'], 3))

# 2.5.1 Linear regression of flow rate on pressure
X = np.array(df.media_pressione_velocita_a_regime).reshape(-1, 1)
y = np.array(df.media_portata_velocita_a_regime).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, color="red")
plt.plot(X, reg.predict(X), color="green")
plt.title("Fase di regime @2300rpm")
plt.xlabel("Pressure [bar]")
plt.ylabel("Flow Rate [L/h]")
plt.savefig(Path(Dict_General["path_data"]).joinpath(("Regime Phase [Pressure, Flow Rate].png")))
plt.close()
print("GP flow rate =", round(reg.intercept_[0], 3), "+", round(reg.coef_[0][0], 3), "* pressure")
print("R^2 =", round(reg.score(X, y), 3))

#####--- 2.6 Analyze Control Phase @140 rpm
print("\nCorrelazione grandezze fase di controllo @140rpm")
print("pressione - portata:\t\t", round(df.corr()['media_pressione_velocita_1']['media_portata_velocita_1'], 3))
print("coppia_zero - pressione:\t", round(df.corr()['media_coppia_zero']['media_pressione_velocita_1'], 3))
print("coppia_zero - portata:\t\t", round(df.corr()['media_coppia_zero']['media_portata_velocita_1'], 3))
print("coppia_finale - pressione:\t", round(df.corr()['media_coppia_finale']['media_pressione_velocita_1'], 3))
print("coppia_finale - portata:\t", round(df.corr()['media_coppia_finale']['media_portata_velocita_1'], 3))

# 2.6.1 Linear regression of flow rate on pressure
X = np.array(df.media_pressione_velocita_1).reshape(-1, 1)
y = np.array(df.media_portata_velocita_1).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, color="red")
plt.plot(X, reg.predict(X), color = "green")
plt.title("Fase di controllo @140rpm")
plt.xlabel("Pressure [bar]")
plt.ylabel("Flow Rate [L/h]")
plt.savefig(Path(Dict_General["path_data"]).joinpath(("Control Phase [Pressure, Flow Rate].png")))
plt.close()
print("GP flow rate =", round(reg.intercept_[0], 3), "+", round(reg.coef_[0][0], 3), "* pressure")
print("R^2 =", round(reg.score(X, y), 3))

#####--- 2.7 Analyze differences between the two phases
# 2.7.1 Linear regression of delta flow rate on delta pressure
X = np.array(df.media_pressione_velocita_a_regime - df.media_pressione_velocita_1).reshape(-1, 1)
y = np.array(df.media_portata_velocita_a_regime - df.media_portata_velocita_1).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, color="red")
plt.plot(X, reg.predict(X), color = "green")
plt.title("\nCambiamento valori da una fase all'altra")
plt.xlabel("Delta Pressure [bar]")
plt.ylabel("Delta Flow Rate [L/h]")
plt.savefig(Path(Dict_General["path_data"]).joinpath(("Delta between phases [Pressure, Flow Rate].png")))
plt.close()
print("\nDelta GP flow rate =", round(reg.intercept_[0], 3), '+', round(reg.coef_[0][0], 3), "* delta pressure")
print("R^2 =", round(reg.score(X, y), 3))

#####--- 2.8 Export preprocessed data
df.to_csv(Path(Dict_General["path_data"]).joinpath("gp5_data.csv"), index_label=False)
print("\nFinished all EDA and preprocessing")