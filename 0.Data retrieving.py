#!/usr/bin/env python
# coding: utf-8

import os, pyodbc
import pandas as pd

#spostare il path su C: per diminuire il tempo di esecuzione
path = r'C:\Users\Mauro\PycharmProjects\DB Bosch'
files = ['caso2_2016.mdb', 'caso2_2017.mdb', 'caso2_2018.mdb', 'caso2_2019.mdb', 'caso2_2020.mdb']
all_data = pd.DataFrame()

for file in files:
    #stabilire connessione tramite il driver ODBC di Microsoft Access
    conn_string = "DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + os.path.join(path, file)
    conn = pyodbc.connect(conn_string)
    
    #ricerca tabella presente (escluse quelle di sistema) e inserimento dell'intero contenuto in un dataframe
    cur = conn.cursor()
    for row in cur.tables():
        if row.table_name.find("MSys")==-1:
            query = "SELECT * FROM " + row.table_name + ";"
            df = pd.read_sql(query, conn)
            
            #scrittura dataframe in un file csv
            name_file_csv = file.split(".")[0] + "_" + row.table_name + ".csv"
            df.to_csv(os.path.join(path, name_file_csv), index_label=False)
            print(file + " " + row.table_name + " scritta in " + name_file_csv + ": " + str(df.shape[0]) + " righe, " + str(df.shape[1]) + " colonne")
            
            #concatenazione record nel dataframe completo
            all_data = pd.concat([all_data, df])
    
    cur.close()
    conn.close()

#scrittura dataframe completo in un file csv
name_file_csv = "caso2_ALL_ESITI.csv"
all_data.to_csv(os.path.join(path, name_file_csv), index_label=False)
print("Dataset completo scritto in " + name_file_csv + ": " + str(all_data.shape[0]) + " righe, " + str(all_data.shape[1]) + " colonne")