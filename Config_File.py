######################### 0. GENERAL PARAMETERS #########################
Dict_General = {
    "path_data": "/Users/ivanmera/Documents/Industry_lab" # To be changed
    }
Dict_EDA_Prepro = {
    "interested_vars": ["picco_coppia_zero", "media_coppia_zero", "picco_pressione_velocita_1",
                         "media_pressione_velocita_1", "picco_portata_velocita_1", "media_portata_velocita_1",
                         "picco_pressione_velocita_a_regime", "media_pressione_velocita_a_regime","picco_coppia_finale",
                         "media_coppia_finale", "velicita_1", "velocita_a_regime", "Temperatura", "n_esito"],
    "n_esito_filter": 100,
    "media_portata_velocita_1_filter_1": 40,
    "media_portata_velocita_1_filter_2": 100,
    }
Dict_Modeling = {
    "dims_clust": ["media_coppia_zero", "media_coppia_finale"],
    "kmax_clust": 10,
    "n_clusters": 2,
    "n_folds_cv": 5
}