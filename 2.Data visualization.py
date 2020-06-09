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
#######################--- 2. MODELING PGM 18_GP5_910_CW.DBF ---########################
#########################################################################################
#####--- 2.0 Split train/test set
df_pgm_18 = df.loc[df["Programma"]=="18_GP5_910_CW.DBF",
                     ["media_portata_velocita_1","media_pressione_velocita_1","Programma"]].copy()
train_18, test_18= train_test_split(df_pgm_18,test_size=0.2)
x_train_18 = train_18.drop(["media_portata_velocita_1","Programma"], 1)
y_train_18 = train_18["media_portata_velocita_1"]
x_test_18 = test_18.drop(["media_portata_velocita_1","Programma"], 1)
y_test_18 = test_18['media_portata_velocita_1']
#####--- 2.0 Linear Regression
lin_reg_18 = LinearRegression()
lin_reg_18.fit(x_train_18, y_train_18)
lr_gp_flow_pred_18 = lin_reg_18.predict(x_test_18)
lr_rmse_18 = np.sqrt(mean_squared_error(y_test_18, lr_gp_flow_pred_18))
lr_r2_18 = r2_score(np.array(y_test_18).reshape(-1, 1),lr_gp_flow_pred_18)
print("\nLR RMSE 18 PGM:", lr_rmse_18)
print("LR R^2 18 PGM:", lr_r2_18)

#####--- 2.1 Decision Tree Regressor
tree_reg_18 = DecisionTreeRegressor()
tree_reg_18.fit(x_train_18, y_train_18)
tree_gp_flow_pred_18 = tree_reg_18.predict(x_test_18)
tree_rmse_18 = np.sqrt(mean_squared_error(y_test_18, tree_gp_flow_pred_18))
tree_r2_18 = r2_score(np.array(y_test_18).reshape(-1, 1),tree_gp_flow_pred_18)
print("\nTREE RMSE 18 PGM:", tree_rmse_18)
print("TREE R^2 18 PGM:", tree_r2_18)

#####--- 2.2 Random Forest Regressor
forest_reg_18 = RandomForestRegressor()
forest_reg_18.fit(x_train_18, y_train_18)
rf_gp_flow_pred_18 = forest_reg_18.predict(x_test_18)
rf_rmse_18 = np.sqrt(mean_squared_error(y_test_18, rf_gp_flow_pred_18))
rf_r2_18 = r2_score(np.array(y_test_18).reshape(-1, 1),rf_gp_flow_pred_18)
print("\nRF RMSE 18 PGM:", rf_rmse_18)
print("RF R^2 18 PGM:", rf_r2_18)

#####--- 2.3 Compute Leakage coefficient
df_pgm_18["gp_theo"] = Dict_Viz["gp_theo"]
df_pgm_18["portata_pred"] = lin_reg_18.intercept_ + float(lin_reg_18.coef_)*df_pgm_18["media_pressione_velocita_1"]
df_pgm_18["leakage_coeff"] = (df_pgm_18["gp_theo"] - df_pgm_18["portata_pred"])/ df_pgm_18["media_pressione_velocita_1"]

#########################################################################################
#######################--- 3. MODELING FOR OTHERS PROGRAMS ---########################
#########################################################################################
#####--- 3.0 Split train/test set
df_pgm_others = df.loc[df["Programma"]!="18_GP5_910_CW.DBF",
                       ["media_portata_velocita_1","media_pressione_velocita_1","Programma"]].copy()
train_pgm_others, test_pgm_others= train_test_split(df_pgm_others,test_size=0.2)
x_train_pgm_others = train_pgm_others.drop(["media_portata_velocita_1","Programma"], 1)
y_train_pgm_others = train_pgm_others["media_portata_velocita_1"]
x_test_pgm_others = test_pgm_others.drop(["media_portata_velocita_1","Programma"], 1)
y_test_pgm_others = test_pgm_others['media_portata_velocita_1']

#####--- 3.1 Linear Regression
lin_reg_pgm_others = LinearRegression()
lin_reg_pgm_others.fit(x_train_pgm_others, y_train_pgm_others)
lr_gp_flow_pred_pgm_others = lin_reg_pgm_others.predict(x_test_pgm_others)
lr_rmse_pgm_others = np.sqrt(mean_squared_error(y_test_pgm_others, lr_gp_flow_pred_pgm_others))
lr_r2_pgm_others = r2_score(np.array(y_test_pgm_others).reshape(-1, 1),lr_gp_flow_pred_pgm_others)
print("\nLR RMSE OTHERS PGM:", lr_rmse_pgm_others)
print("LR R^2 OTHERS PGM:", lr_r2_pgm_others)

#####--- 3.2 Decision Tree Regressor
tree_reg_pgm_others = DecisionTreeRegressor()
tree_reg_pgm_others.fit(x_train_pgm_others, y_train_pgm_others)
tree_gp_flow_pred_pgm_others = tree_reg_pgm_others.predict(x_test_pgm_others)
tree_rmse_pgm_others = np.sqrt(mean_squared_error(y_test_pgm_others, tree_gp_flow_pred_pgm_others))
tree_r2_pgm_others = r2_score(np.array(y_test_pgm_others).reshape(-1, 1),tree_gp_flow_pred_pgm_others)
print("\nTREE RMSE OTHERS PGM:", tree_rmse_pgm_others)
print("TREE R^2 OTHERS PGM:", tree_r2_pgm_others)

#####--- 3.3 Random Forest Regressor
forest_reg_pgm_others = RandomForestRegressor()
forest_reg_pgm_others.fit(x_train_pgm_others, y_train_pgm_others)
rf_gp_flow_pred_pgm_others = forest_reg_pgm_others.predict(x_test_pgm_others)
rf_rmse_pgm_others = np.sqrt(mean_squared_error(y_test_pgm_others, rf_gp_flow_pred_pgm_others))
rf_r2_pgm_others = r2_score(np.array(y_test_pgm_others).reshape(-1, 1),rf_gp_flow_pred_pgm_others)
print("\nRF RMSE OTHERS PGM:", rf_rmse_pgm_others)
print("RF R^2 OTHERS PGM:", rf_r2_pgm_others)

#####--- 3.4 Compute Leakage coefficient
df_pgm_others["gp_theo"] = Dict_Viz["gp_theo"]
df_pgm_others["portata_pred"] = lin_reg_pgm_others.intercept_ + float(lin_reg_pgm_others.coef_)*df_pgm_others["media_pressione_velocita_1"]
df_pgm_others["leakage_coeff"] = (df_pgm_others["gp_theo"] - df_pgm_others["portata_pred"])/ df_pgm_others["media_pressione_velocita_1"]



#########################################################################################
#######################--- 4. MERGE AND EXPORT DATA---########################
#########################################################################################
df_final = pd.concat([df_pgm_18,df_pgm_others])
#df_final.to_excel(Path(Dict_General["path_data"]).joinpath("gp5_viz_data.xlsx"), index = False)
df_final.to_csv(Path(Dict_General["path_data"]).joinpath("gp5_viz_data.csv"),
                index = False, sep = ";", decimal = ",")
print("\nExported all files")
