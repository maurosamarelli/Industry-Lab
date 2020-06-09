#########################################################################################
#########################--- 0. IMPORT LIBRARIES & PACKAGES---###########################
#########################################################################################
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from Config_File import Dict_General
from Config_File import Dict_Viz
#########################################################################################
#######################--- 1. IMPORT DATA & SET-UP PARAMETERS/ Functions ---########################
#########################################################################################
#####--- 1.0 Import dataset
df = pd.read_csv(Path(Dict_General["path_data"]).joinpath("gp5_data.csv"))
print("Dataset dimension:", df.shape)

#########################################################################################
#######################--- 2. MODELING---########################
#########################################################################################
#####--- 1.0 Split train/test set
df_mod = df[["media_portata_velocita_1","media_pressione_velocita_1"]]
train, test= train_test_split(df_mod,test_size=0.2)
x_train = train.drop("media_portata_velocita_1", 1)
y_train = train["media_portata_velocita_1"]
x_test = test.drop('media_portata_velocita_1', 1)
y_test = test['media_portata_velocita_1']

#####--- 2.0 Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lr_gp_flow_pred = lin_reg.predict(x_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_gp_flow_pred))
lr_r2 = r2_score(np.array(y_test).reshape(-1, 1),lr_gp_flow_pred)
print("\nLR RMSE:", lr_rmse)
print("LR R^2:", lr_r2)

#####--- 2.1 Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
tree_gp_flow_pred = tree_reg.predict(x_test)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_gp_flow_pred))
tree_r2 = r2_score(np.array(y_test).reshape(-1, 1),tree_gp_flow_pred)
print("\nTREE RMSE:", tree_rmse)
print("TREE R^2:", tree_r2)

#####--- 2.2 Random Forest Regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(x_train, y_train)
rf_gp_flow_pred = forest_reg.predict(x_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_gp_flow_pred))
rf_r2 = r2_score(np.array(y_test).reshape(-1, 1),rf_gp_flow_pred)
print("\nRF RMSE:", rf_rmse)
print("RF R^2:", rf_r2)

#####--- 2.3 Compute Leakage coefficient
df["gp_theo"] = Dict_Viz["gp_theo"]
df["portata_pred"] = lin_reg.intercept_ + float(lin_reg.coef_)*df["media_pressione_velocita_1"]
df["leakage_coeff"] = (df["gp_theo"] - df["portata_pred"])/ df["media_pressione_velocita_1"]

#########################################################################################
#######################--- 3. EXPORT DATA---########################
#########################################################################################
df.to_excel(Path(Dict_General["path_data"]).joinpath("gp5_viz_data.xlsx"), index = False)
print("\nExported all files")

#riga di prova
