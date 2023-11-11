import pandas as pd

def dataset_load():
    caminho_dados = 'risco_credito.csv'
    df = pd.read_csv(caminho_dados)
    return df
