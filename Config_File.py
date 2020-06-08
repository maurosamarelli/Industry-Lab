######################### 0. GENERAL PARAMETERS #########################
Dict_General = {
    "path_data": "/Users/ivanmera/Documents/Industry_Lab" # To be changed
    }
Dict_EDA_Prepro = {
    "interested_vars": ['picco_coppia_zero', 'media_coppia_zero', 'picco_pressione_velocita_1', 'media_pressione_velocita_1',
                   'picco_portata_velocita_1', 'media_portata_velocita_1', 'picco_pressione_velocita_a_regime',
                   'media_pressione_velocita_a_regime', 'picco_portata_velocita_a_regime', 'media_portata_velocita_a_regime',
                   'picco_coppia_finale', 'media_coppia_finale', 'velicita_1', 'velocita_a_regime', 'Temperatura', 'n_esito'],
    "filter_n_esito": "100",
    "filter_velocita_1": "140",
    "filter_velocita_a_regime": "2300",
    "filter_min_temperatura": "39",
    "filter_max_temperatura": "43",
    "filter_positive_values": "media_portata_velocita_a_regime > 0 and media_pressione_velocita_a_regime > 0" +
                              " and media_portata_velocita_1 > 0 and media_pressione_velocita_1 > 0",
    "n_obs_per_pgm": 1000
}
Dict_Viz = {
    "gp_theo": 108.36
}