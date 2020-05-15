#########################################################################################
#########################--- 0. IMPORT LIBRARIES & PACKAGES---###########################
#########################################################################################
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Config_File import Dict_General
from Config_File import Dict_Modeling
from sklearn.linear_model import LinearRegression

#########################################################################################
#######################--- 1. IMPORT DATA & SET-UP PARAMETERS/ Functions ---########################
#########################################################################################
#####--- 1.0 Create a function
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)
    return sse

#####--- 1.1 Import dataset
df = pd.read_csv(Path(Dict_General["path_data"]).joinpath("caso_2_after_eda_prepro.csv"))

#########################################################################################
#######################--- 2. Linear regression ---########################
#########################################################################################
#####--- 2.0 Linear regression model
# 2.0.0 Create LR model
X = np.array(df.media_pressione_velocita_a_regime).reshape(-1, 1)
y = np.array(df.media_portata_velocita_1).reshape(-1, 1)
lin_reg = LinearRegression();
lin_reg.fit(X,y)

# 2.0.1 Create scatterplot
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "green")
plt.title("Plot")
plt.xlabel("Pressure")
plt.ylabel("Flow Rate")
plt.savefig(Path(Dict_General["path_data"]).joinpath("Linear_reg.png"))
print('R squared: ',"R^2 =", lin_reg.score(X, y))
print('GP theoretical flow rate: ', lin_reg.intercept_[0])
print('Leakage coefficient: ', lin_reg.coef_[0][0])

#########################################################################################
#######################--- 3. Clustering ---########################
#########################################################################################
#####--- 3.0 Create a scatterplot
plt.scatter(df.media_coppia_zero, df.media_coppia_finale)
plt.title("Coppia")
plt.xlabel("media coppia zero [Nm]")
plt.ylabel("media coppia finale [Nm]")
plt.savefig(Path(Dict_General["path_data"]).joinpath("Coppia_correlation.png"))

#####--- 3.1 Calculate WSS
X = np.array(df[Dict_Modeling["dims_clust"]])
res = calculate_WSS(X, Dict_Modeling["kmax_clust"])
plt.scatter(list(np.arange(Dict_Modeling["kmax_clust"])), res, color = "red")
plt.savefig(Path(Dict_General["path_data"]).joinpath("WSS.png"))
print(res)

#####--- 3.2 Create cluster
X = np.array(df[Dict_Modeling["dims_clust"]])
kmeans = KMeans(n_clusters = Dict_Modeling["n_clusters"], random_state = 0).fit(X)
print(kmeans.cluster_centers_)
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts/len(df))))

#####--- 3.3 Create cluster graph
df['cluster'] = kmeans.labels_
for i in range(Dict_Modeling["n_clusters"]):
    plt.scatter(df.query("cluster == " + str(i)).picco_pressione_velocita_a_regime, df.query("cluster == " + str(i)).picco_portata_velocita_1)
    plt.title("cluster=" + str(i) + ", speed=140rpm, T=40°C")
    plt.xlabel("outlet pressure [bar]")
    plt.ylabel("GP flow rate [L/h]")
    plt.savefig(Path(Dict_General["path_data"]).joinpath("Cluster.png"))

#########################################################################################
#######################--- 4. Leakage prediction ---########################
#########################################################################################
#####--- 4.0 Prediction with LR
cluster0 = df.query("cluster == 0")
cluster1 = df.query("cluster == 1")
clusters = [cluster0, cluster1]
n_folds, fold = Dict_Modeling["n_folds_cv"], []
cluster, test_fold, r2, gp_flow_rate, leakage_coeff, test_mae, test_mse = [], [], [], [], [], [], []
models = {}

for c in clusters:
    for i in range(len(c)):
        fold.append(i % n_folds)
    c['fold'] = pd.Series(fold)
    for i in range(n_folds):
        X_train = np.array(c.query("fold != " + str(i)).media_pressione_velocita_a_regime).reshape(-1, 1)
        y_train = np.array(c.query("fold != " + str(i)).media_portata_velocita_1).reshape(-1, 1)
        X_test = np.array(c.query("fold == " + str(i)).media_pressione_velocita_a_regime).reshape(-1, 1)
        y_test = np.array(c.query("fold == " + str(i)).media_portata_velocita_1).reshape(-1, 1)
        lin_reg = LinearRegression().fit(X_train, y_train)
        cluster.append(c.cluster.unique()[0])
        test_fold.append(i)
        r2.append(lin_reg.score(X_train, y_train))
        gp_flow_rate.append(lin_reg.intercept_[0])
        leakage_coeff.append(lin_reg.coef_[0][0])
        test_mae.append(mean_absolute_error(y_test, lin_reg.predict(X_test)))
        test_mse.append(mean_squared_error(y_test, lin_reg.predict(X_test)))
        models[(c.cluster.unique()[0], i)] = lin_reg

results = pd.DataFrame({'cluster':pd.Series(cluster), 'test_fold':pd.Series(test_fold), 'r2':pd.Series(r2),
                        'theoretical_gp_flow_rate':pd.Series(gp_flow_rate), 'leakage_coeff':pd.Series(leakage_coeff),
                        'test_mae':pd.Series(test_mae), 'test_mse':pd.Series(test_mse)})
print(results)

#####--- 4.1 Plot best model
# 4.1.0 Cluster 0
test_fold_n=results.groupby(['cluster'])['test_mse'].idxmin().values
lr = models[(0, test_fold_n[0])]
X = np.array(cluster0.query("fold != 3")['media_pressione_velocita_a_regime']).reshape(-1, 1)
y = np.array(cluster0.query("fold != 3")['media_portata_velocita_1']).reshape(-1, 1)
plt.scatter(X, y, color = "red")
plt.plot(X, lr.predict(X), color = "green")
plt.title("Cluster 0")
plt.xlabel("Pressure")
plt.ylabel("Flow Rate")
plt.savefig(Path(Dict_General["path_data"]).joinpath("best_model_cluster_0.png"))
print(results[(results["cluster"] == 0) & (results["test_fold"] == test_fold_n[0])])

# 4.1.1 Cluster 1
lr = models[(1, test_fold_n[1])]
X = np.array(cluster1.query("fold != 1")['media_pressione_velocita_a_regime']).reshape(-1, 1)
y = np.array(cluster1.query("fold != 1")['media_portata_velocita_1']).reshape(-1, 1)
plt.scatter(X, y, color = "red")
plt.plot(X, lr.predict(X), color = "green")
plt.title("Cluster 1")
plt.xlabel("Pressure")
plt.ylabel("Flow Rate")
plt.savefig(Path(Dict_General["path_data"]).joinpath("Best_model_cluster_1.png"))
print(results[(results["cluster"] == 1) & (results["test_fold"] == test_fold_n[1])])

#####--- 4.2 Export cluster
c0 = cluster0.query("fold != 3")[['media_pressione_velocita_a_regime', 'media_portata_velocita_1']]
c1 = cluster1.query("fold != 1")[['media_pressione_velocita_a_regime', 'media_portata_velocita_1']]
c0.to_csv(Path(Dict_General["path_data"]).joinpath("cluster0.csv"), index_label=False)
c1.to_csv(Path(Dict_General["path_data"]).joinpath("cluster1.csv"), index_label=False)

## sjsjsdjsdjdndsjsd