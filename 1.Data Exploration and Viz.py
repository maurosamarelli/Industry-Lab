#########################################################################################
#########################--- 0. IMPORT LIBRARIES & PACKAGES---###########################
#########################################################################################
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from Config_File import Dict_General,Dict_EDA_Prepro,Dict_Viz
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter("ignore")

#########################################################################################
#######################--- 1. IMPORT DATA & SET-UP PARAMETERS ---########################
#########################################################################################
df = pd.read_csv(Path(Dict_General["path_data"]).joinpath("caso2_ALL_ESITI.csv"))
df.reset_index(inplace = True)
df.drop("index",axis = 1, inplace = True)

#########################################################################################
#######################--- 2. EXPLORATORY DATA ANALYSIS ---########################
#########################################################################################
#####--- 2.0 Print some dataset info
print("Dataset dimension: ", df.shape)
print("\nDataset dtypes: ",df.dtypes)
print("n_esito:", df["n_esito"].value_counts())

#####--- 2.1 Plot numeric interested variables
df[Dict_EDA_Prepro["var_dist"]].hist(bins=100, figsize=(20,15))
plt.savefig(Path(Dict_General["path_img_out"]).joinpath(("Variables_distribution.png")))
plt.close()

#####--- 2.2 Analyze Regime Phase @2300 rpm correlations
print("\nCorrelazione grandezze fase di regime @2300rpm")

# 2.2.0 Pressione - Portata
print("pressione - portata:\t\t", round(df.corr()['media_pressione_velocita_a_regime']['media_portata_velocita_a_regime'], 3))

# 2.2.1 Coppia_zero - Pressione
print("coppia_zero - pressione:\t", round(df.corr()['media_coppia_zero']['media_pressione_velocita_a_regime'], 3))

# 2.2.2 Coppia_zero - Portata
print("coppia_zero - portata:\t\t", round(df.corr()['media_coppia_zero']['media_portata_velocita_a_regime'], 3))

# 2.2.3 Coppia_finale - Pressione
print("coppia_finale - pressione:\t", round(df.corr()['media_coppia_finale']['media_pressione_velocita_a_regime'], 3))

# 2.2.4 Coppia_finale - Portata
print("coppia_finale - portata:\t", round(df.corr()['media_coppia_finale']['media_portata_velocita_a_regime'], 3))

# 2.2.5 Save correlation graph
df_for_corr_graph = df[["media_pressione_velocita_a_regime","media_pressione_velocita_1",
                        "media_portata_velocita_a_regime", "media_portata_velocita_1",
                        "media_coppia_zero","media_coppia_finale","Temperatura"]]
df_for_corr_graph.rename(columns={"media_pressione_velocita_a_regime":"Pressione_Regime",
                                  "media_pressione_velocita_1": "Pressione_Controllo",
                                  "media_portata_velocita_a_regime":"Portata_Regime",
                                  "media_portata_velocita_1":"Portata_Controllo",
                                  "media_coppia_zero":"Coppia_Zero",
                                  "media_coppia_finale":"Coppia_Finale"}, inplace = True)
#corr_regime_phase = pd.plotting.scatter_matrix(df_for_corr_graph[["Pressione_Regime","Portata_Regime",
                                                                  #"Coppia_Zero","Coppia_Finale","Temperatura"]],
                                               #figsize=(16, 12), diagonal = "kde")
#[s.xaxis.label.set_rotation(45) for s in corr_regime_phase.reshape(-1)]
#[s.yaxis.label.set_rotation(0) for s in corr_regime_phase.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
#[s.get_yaxis().set_label_coords(-0.3,0.5) for s in corr_regime_phase.reshape(-1)]
#Hide all ticks
#[s.set_xticks(()) for s in corr_regime_phase.reshape(-1)]
#[s.set_yticks(()) for s in corr_regime_phase.reshape(-1)]
#plt.savefig(Path(Dict_General["path_img_out"]).joinpath(("Correlation_Regime_Phase.png")))
#plt.close()
#####--- 2.3 Analyze Control Phase @140 rpm
print("\nCorrelazione grandezze fase di controllo @140rpm")

# 2.3.0  Pressione - Portata
print("pressione - portata:\t\t", round(df.corr()['media_pressione_velocita_1']['media_portata_velocita_1'], 3))

# 2.3.1  Coppia_zero - Pressione
print("coppia_zero - pressione:\t", round(df.corr()['media_coppia_zero']['media_pressione_velocita_1'], 3))

# 2.3.2 Coppia_zero - Portata
print("coppia_zero - portata:\t\t", round(df.corr()['media_coppia_zero']['media_portata_velocita_1'], 3))

# 2.3.3 Coppia_finale - Pressione
print("coppia_finale - pressione:\t", round(df.corr()['media_coppia_finale']['media_pressione_velocita_1'], 3))

# 2.3.4 Coppia_finale - Portata
print("coppia_finale - portata:\t", round(df.corr()['media_coppia_finale']['media_portata_velocita_1'], 3))

# 2.3.5 Save correlation graph
#corr_control_phase = pd.plotting.scatter_matrix(df_for_corr_graph[["Pressione_Controllo","Portata_Controllo",
                                                                  #"Coppia_Zero","Coppia_Finale","Temperatura"]],
                                                #figsize=(16, 12), diagonal = "kde")
#[s.xaxis.label.set_rotation(45) for s in corr_control_phase.reshape(-1)]
#[s.yaxis.label.set_rotation(0) for s in corr_control_phase.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
#[s.get_yaxis().set_label_coords(-0.3,0.5) for s in corr_control_phase.reshape(-1)]
#Hide all ticks
#[s.set_xticks(()) for s in corr_control_phase.reshape(-1)]
#[s.set_yticks(()) for s in corr_control_phase.reshape(-1)]
#plt.savefig(Path(Dict_General["path_img_out"]).joinpath(("Correlation_Control_Phase.png")))
#plt.close()

#####--- 2.4 Analyze differences between the two phases
# 2.4.0 Linear regression of delta flow rate on delta pressure
X = np.array(df.media_pressione_velocita_a_regime - df.media_pressione_velocita_1).reshape(-1, 1)
y = np.array(df.media_portata_velocita_a_regime - df.media_portata_velocita_1).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, color="blue")
plt.plot(X, reg.predict(X), color = "red")
plt.title("\nCambiamento valori da una fase all'altra")
plt.xlabel("Delta Pressure [bar]")
plt.ylabel("Delta Flow Rate [L/h]")
plt.savefig(Path(Dict_General["path_img_out"]).joinpath(("Delta between phases [Pressure, Flow Rate].png")))
plt.close()
print("\nDelta GP flow rate =", round(reg.intercept_[0], 3), '+', round(reg.coef_[0][0], 3), "* delta pressure")
print("R^2 =", round(reg.score(X, y), 3))

#####--- 2.5  Filter Dataset for a good evaluation
# 2.5.0 Apply some filters
df = df.query("n_esito == " + Dict_EDA_Prepro["filter_n_esito"] +
              " and velicita_1 == " + Dict_EDA_Prepro["filter_velocita_1"] +
              " and velocita_a_regime == " + Dict_EDA_Prepro["filter_velocita_a_regime"] +
              " and " + Dict_EDA_Prepro["filter_positive_values"])

# 2.5.1 Remove "Programmma" with less than 1000 osservations
print(df["Programma"].value_counts())
df = df.groupby('Programma').filter(lambda x: len(x) >=Dict_EDA_Prepro["n_obs_per_pgm"])
print(len(df[df["n_esito"]!=100]))

#####--- 2.6 Look for missing values
# 2.6.1
print("\nMissing values per variable: ")
print(df.isnull().sum())

# 2.6.2 Fill missing values
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=' ')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
print("Numero missing values:", sum(df.isnull().sum()))

#####--- 2.7 Remove univariate outliers
print("\nDataframe dimension with outliers: ", len(df))

# 2.7.0 Tranform column dtype
for var in Dict_EDA_Prepro["interested_vars"]:
    if var != "n_esito":
        df[var] = df[var].astype(float)

# 2.7.1 Remove outliers
for var in Dict_EDA_Prepro["interested_vars"]:
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
    if var != "n_esito":
        if df[var].std() != 0:
            df['zscore'] = pd.Series(np.abs(zscore(df[var])))
            idx = list(df.query("zscore > 3.5").index)
            print(var, len(idx))
            df.drop(idx, inplace=True)
            df.drop("zscore", 1, inplace=True)

print("Dataframe dimension without outliers: ", df.shape)
#########################################################################################
#######################--- 3. VISUALIZATION ---########################
#########################################################################################
#####--- 3.0 Compute Leakage coefficient
df["gp_theo"] = Dict_Viz["gp_theo"]
df["leakage_coeff"] = (df["gp_theo"] - df["media_portata_velocita_1"])/ df["media_pressione_velocita_1"]

#########################################################################################
#######################--- 4. EXPORT DATA---########################
#########################################################################################
df.to_csv(Path(Dict_General["path_data"]).joinpath("gp5_viz_data.csv"),
                index = False, sep = ";", decimal = ",")
print("\nExported all files")