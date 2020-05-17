#########################################################################################
#########################--- 0. IMPORT LIBRARIES & PACKAGES---###########################
#########################################################################################
from pathlib import Path
import pandas as pd
from Config_File import Dict_General
from Config_File import Dict_Modeling

#########################################################################################
#######################--- 1. IMPORT DATA & SET-UP PARAMETERS/ Functions ---########################
#########################################################################################
#####--- 1.1 Import dataset
df = pd.read_csv(Path(Dict_General["path_data"]).joinpath("gp5_data.csv"))
print("Dimensioni dataset dopo preprocessing:", df.shape)

#########################################################################################
#######################--- 2. Leakage Coefficient ---########################
#########################################################################################
#####--- 2.0 Fisso efficienza a regime (ipotesi)
df['eff_regime'] = Dict_Modeling["eff_regime"]
print("Fisso efficienza a regime:", Dict_Modeling["eff_regime"])

#####--- 2.1 Ricavo efficienza e portata teorica sia a regime sia a velocità 140 rpm
##### equazione: alfa_140 = alfa_regime
##### equazione: (portata_teorica_140 - portata_140) / eff_140 = (portata_teorica_regime - portata_regime) / eff_regime

##### eff_regime = portata_regime / portata_teorica_regime --> portata_teorica_regime = portata_regime / eff_regime
##### eff_140 = portata_140 / portata_teorica_140          --> portata_teorica_140 = portata_140 / eff_140
##### K = eff_140 / eff_regime
##### (spiegazione:  https://www.machinerylubrication.com/Read/28430/hydraulic-pump-motors-maintenance)

##### sostituendo le formule nell'equazione si ottiene:
##### K = A / (1 + eff_regime * (A - 1))
##### con A = (portata_140 pressione_regime) / (pressione_140 portata_regime)
##### per ogni riga si ottiene efficienza e portata teorica a 140 rpm (fissando efficienza e portata teorica a regime)
A = (df.media_portata_velocita_1 * df.media_pressione_velocita_a_regime) / (df.media_pressione_velocita_1 * df.media_portata_velocita_a_regime)
df['eff_1'] = (df.eff_regime * A) / (1 + df.eff_regime * (A - 1))
print("\nDistribuzione efficienza a velocità 140 rpm")
print(df.eff_1.describe())

df['portata_teorica_velocita_1'] = df.media_portata_velocita_1 / df.eff_1
print("\nDistribuzione portata teorica a velocità 140 rpm")
print(df.portata_teorica_velocita_1.describe())

df['portata_teorica_velocita_a_regime'] = df.media_portata_velocita_a_regime / df.eff_regime
print("\nDistribuzione portata teorica a velocità a regime")
print(df.portata_teorica_velocita_a_regime.describe())

#####--- 2.2 Ricavo il valore di alfa (leakage coefficient) che coincide tra regime e velocità 140 rpm
##### A questo punto sappiamo tutto per calcolare alfa_140 che coincide con alfa_regime (dipende da caratteristiche interne)
##### Ciò che cambia da 140 rpm a 2300 rpm è l'efficienza volumetrica non il coefficiente di perdita
##### alfa_140 = (portata_teorica_140 - portata_140) / pressione_140
##### alfa_regime = (portata_teorica_regime - portata_regime) / pressione_regime

alfa_1 = (df.portata_teorica_velocita_1 - df.media_portata_velocita_1) / df.media_pressione_velocita_1
print("\nDistribuzione alfa a velocità a 140 rpm")
print(alfa_1.describe())

alfa_regime = (df.portata_teorica_velocita_a_regime - df.media_portata_velocita_a_regime) / df.media_pressione_velocita_a_regime
print("\nDistribuzione alfa a velocità a regime")
print(alfa_regime.describe())
print("\nVerificare che la distribuzione valori di alfa a 140 rpm sia uguale a quella di alfa a regime!")

#####--- 2.3 Export dataset per regressione e/o clustering
df['alfa'] = alfa_1
print("\nDimensioni dataset per regressione e/o clustering:", df.shape)
df.to_csv(Path(Dict_General["path_data"]).joinpath("final_data.csv"), index_label=False)