import pandas as pd

def dataset_load():
    caminho_dados = 'data\creditcard1.csv'
    df = pd.read_csv(caminho_dados, compression="gzip")
    return df
