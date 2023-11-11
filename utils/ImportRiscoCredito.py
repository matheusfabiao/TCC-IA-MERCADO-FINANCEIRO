# Importando bibliotecas
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score
import streamlit as st
from imblearn.over_sampling import SMOTE
# import pickle
from utils.Global import *
from data.risco_credito import dataset_load

import warnings
warnings.filterwarnings('ignore')


def titulo_app():
    # st.write("*TCC - Trabalho de Conclusão de Curso*")
    # st.write("*Deploy de Modelo Preditivo de Machine Learning*")
    # st.write("*Previsão do Risco de Crédito*")
    # st.write("*Autor: Matheus Fabião da Costa Pereira*")
    # st.write("*Autor: Matheus Davi de Serrano Araújo Meireles*")
    # st.write("*Autor: Leandro Santana de Melo*")
    st.title("Random Forest Classifier Model")


@st.cache_data
def carrega_dataset():
    # Carregando os dados
    try:
        dados = dataset_load()
        return dados
    except Exception as e:
        st.error(f'Erro ao carregar dados: {str(e)}')
        return None


def pre_processar_dados(dados):
    dados.drop(columns=['Unnamed: 0'], axis=1, inplace=True) 
    dados["Saving accounts"] = dados["Saving accounts"].fillna("none") 
    dados["Checking account"] = dados["Checking account"].fillna("none")
    return dados


def exibe_dataset(dados):
    # Exibir os últimos 5 dados do Dataset
    pre_processar_dados(dados)
    st.subheader('Dataset: Risco de Crédito')
    st.dataframe(dados)
    
    
def info_dataset():
    # Exibe um expander ao clicar no botão de ajuda
    with st.expander("ℹ️ **Informações do Dataset**"):
        st.write('''**Age:** Idade do cliente.             
             **Sex:** Gênero do cliente (Masculino ou Feminino).            
             **Job:** Tipo de ocupação do cliente.            
             **Housing:** Tipo de moradia do cliente (Própria, Gratuita ou Alugada).            
             **Saving accounts:** Estado da conta de poupança do cliente.            
             **Checking account:** Estado da conta corrente do cliente.            
             **Credit amount:** Valor do crédito solicitado/concedido.            
             **Duration:** Duração do empréstimo em meses.             
             **Purpose:** Finalidade do empréstimo.            
             **Risk:** Risco de crédito associado à solicitação (Bom ou Ruim).''')
    

def codifica_dados(dados):
    # Separe as colunas categóricas
    colunas_categoricas = dados.select_dtypes(exclude=['int64', 'float64'])
    
    le = LabelEncoder()
    for coluna in colunas_categoricas:
        dados[coluna]=le.fit_transform(dados[coluna])
        
    return dados


def separa_dados(dados, split):
    # Separando as características (x) e as classes alvo (y) dos dados
    x = dados.drop(['Risk'], axis=1)
    y = dados['Risk']
    
    smote = SMOTE(random_state=42)
    x_reamostrado, y_reamostrado = smote.fit_resample(x, y)
    
    x_treino, x_teste, y_treino, y_teste = train_test_split(x_reamostrado, y_reamostrado, train_size=split, random_state=42)
    
    return x_treino, x_teste, y_treino, y_teste


def declara_normalizador():
    Normalizador = StandardScaler()
    return Normalizador


def normaliza_dados(x, y, Normalizador):
    # Normalização dos dados
    x_treino = Normalizador.fit_transform(x)
    x_teste = Normalizador.transform(y)
    
    return x_treino, x_teste


def cria_modelo(Hiperparametros):
    # Cria o modelo de RFC
    modelo = RandomForestClassifier(n_estimators= Hiperparametros['N_estimators'],
                                             max_depth= Hiperparametros['Max_depth'],
                                             min_samples_split= Hiperparametros['Min_samples_split'],
                                             class_weight= Hiperparametros['Class_weight'])    
    return modelo


def treina_modelo(modelo, Xtreino, Ytreino):
    modelo.fit(Xtreino, Ytreino)
    

def faz_previsao(modelo, Xteste):
    previsao = modelo.predict(Xteste)   
    return previsao


# Função para calcular métricas de desempenho
def calcula_metricas(Yteste, Previsao):
    # Avaliando o modelo usando as principais métricas
    precisao = precision_score(Yteste, Previsao)
    recall = recall_score(Yteste, Previsao)
    f1 = f1_score(Yteste, Previsao)
    acuracia = accuracy_score(Yteste, Previsao)
    matriz_confusao = confusion_matrix(Yteste, Previsao)
    return precisao, recall, f1, acuracia, matriz_confusao


# Função para exibir métricas de desempenho
def exibe_metricas(Precisao, Recall, F1, Acuracia, Matriz_confusao):
        # Avaliando o modelo usando as principais métricas
        col1, col2 = st.columns(2)
        col1.write(f'Precisão: {Precisao:.2f}')
        col1.write(f'Recall: {Recall:.2f}')
        col1.write(f'F1-Score: {F1:.2f}')
        col1.write(f'Acurácia: {Acuracia:.2f}')
        col2.subheader('Matriz de Confusão:')
        col2.write(Matriz_confusao)


def ajuda_config():
    # Adiciona um botão na sidebar
    if st.sidebar.button('Ajuda com os Hiperparâmetros'):
        # Exibe um expander ao clicar no botão de ajuda
        with st.expander("ℹ️ **Informações de Ajuda**"):
            st.write('''
                    **Intervalo de Tempo:** quantidade total de dados do ativo nos últimos 'n' anos.
                    
                    **Divisão dos Dados:** dividir o total de dados em x% para treino da IA. Recomendado entre 0.7 e 0.8

                    **Dropout:** desativa aleatoriamente neurônios durante o treinamento para evitar *overfitting*.
                    Recomendado: entre 0.2 e 0.5
 
                    **Número de Amostras:** número de exemplos de dados que a IA analisa de uma vez durante o treinamento.
                    Quanto mais alto mais rápido o treinamento, porém exigirá mais do seu hardware.
 
                    **Janela de Entrada:** é como uma "memória" que a rede utiliza para processar sequências de dados, lembrando de informações importantes ao longo do tempo.
 
                    **Épocas:** quantidade de vezes que a IA percorre todo o conjunto de dados durante o treinamento. Quanto mais alto mais demorado o treinamento, porém,
                    mais bem treinada ela será.
                    ''')


def normaliza_novos_dados(Novos_dados, Normalizador, Nome):
    # Criando o DataFrame
    df = pd.DataFrame(Novos_dados)

    # Exibindo o DataFrame
    st.info(f'Exibindo dados de {Nome}')
    st.dataframe(df)
    
    # Separe as colunas categóricas
    colunas_categoricas = df.select_dtypes(exclude=['int64', 'float64'])
    
    le = LabelEncoder()
    for coluna in colunas_categoricas:
        df[coluna]=le.fit_transform(df[coluna])
        
    # x = np.array(Novos_dados).reshape(1, -1)
    # Transforma o array 1D em 2D
    x = Normalizador.transform(df)
    return x


def previsao_novos_dados(modelo, X):
    good = 'Baixo Risco'
    bad = 'Alto Risco'
    previsao = modelo.predict(X)
    if previsao[0] == 1:
        st.write('O resultado da previsão foi:', bad)
    else:
        st.write('O resultado da previsão foi:', good)
    return previsao