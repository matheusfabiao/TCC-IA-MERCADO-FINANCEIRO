# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')

# Importando bibliotecas necessárias
import time
import streamlit as st
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc

# Código de Detecção de Fraudes em Transações de Cartão de Crédito
# Autor: Matheus Fabião da Costa Pereira
# Data de Criação: 30/10/2023

# Função para exibir o título do aplicativo
def titulo_app():
    st.write("TCC - Trabalho de Conclusão de Curso")
    st.write("Deploy de Modelo Preditivo de Machine Learning")
    st.write("Previsão do Preço de Fechamento das Ações")
    st.write("Autor: Matheus Fabião da Costa Pereira")
    st.write("Autor: Matheus Davi de Serrano Araújo Meireles")
    st.write("Orientador: Leandro Santana de Melo")
    st.title("Logistic Regression Model")
    

# Função para carregar o dataset
@st.cache_data
def carrega_dataset():
    caminho_dados = r'C:\Users\Matheus Fabiao\Desktop\TCC-IA-MERCADO-FINANCEIRO\data\creditcard.csv'
    try:
        dados = pd.read_csv(caminho_dados)
        return dados
    except Exception as e:
        st.error(f'Erro ao carregar dataset: {str(e)}')
        return None


# Função para exibir o dataset
def exibe_dataset(dados) -> None:
    # Exibir os últimos 5 dados do Dataset
    st.subheader('Dataset: Detecção de Fraudes')
    st.dataframe(dados)


# Função para exibir informações sobre o dataset
def info_dataset():
    st.write('''**Time:** O tempo decorrido desde a primeira transação no conjunto de dados, medido em segundos.             
             **V1-V28:** Recursos anonimizados que representam várias características da transação (por exemplo, hora, localização, etc.).             
             **Amount:** O valor da transação.             
             **Class:** Rótulo binário que indica se a transação é fraudulenta (1) ou não (0).''')


# Função para remover a coluna 'Time' dos dados
def remove_tempo(dados):
    # Removendo a coluna 'Time', por não ser relevante
    dados = dados.drop(['Time'], axis=1)
    return dados


# Função para separar transações em fraudes e não fraudes
def separa_fraudes(dados):
    # Separando as transações em não fraudes
    nao_fraudes = dados[dados['Class']==0]
    st.write('O número de transações válidas é de:', len(nao_fraudes))

    # Separando as transações em fraudes
    fraudes = dados[dados['Class']==1]
    st.write('O número de transações fraudulentas é de:', len(fraudes))
    
    return nao_fraudes, fraudes


# Função para criar um conjunto de validação
def amostra_validacao(nao_fraudes, fraudes, dados):
    # Criando um conjunto de validação com 15 transações de cada classe
    validacao_nao_fraudes = nao_fraudes.sample(15)
    validacao_fraudes = fraudes.sample(15)
    # Concatenando os dois conjuntos de validação
    validacao = pd.concat([validacao_nao_fraudes, validacao_fraudes], ignore_index=True)
    # Removendo as transações de validação do DataFrame original
    dados = dados.drop(validacao_nao_fraudes.index)
    dados = dados.drop(validacao_fraudes.index)
    # Criando o DataFrame com os valores reais, para comparar com as previsões
    validacao_real = validacao.Class
    # Remover a coluna alvo 'Class'
    validacao = validacao.drop(['Class'], axis=1)
    return validacao, dados, validacao_real


# Função para preparar os dados para treinamento
def prepara_dados(dados):
    # Separando as características (x) e as classes alvo (y) dos dados
    X = dados.drop(['Class'], axis=1)
    Y = dados['Class']
    
    # Utilização do SMOTE para a sobreamostragem
    smote = SMOTE(random_state=42)
    X_reamostrado, y_reamostrado = smote.fit_resample(X, Y)
    return X_reamostrado, y_reamostrado


# Função para separar os dados em conjuntos de treinamento e teste
def separa_dados(X_reamostrado, y_reamostrado, split):  
    x_treino, x_teste, y_treino, y_teste = train_test_split(X_reamostrado, y_reamostrado, train_size=split, random_state=42, stratify=y_reamostrado)
    return x_treino, x_teste, y_treino, y_teste


# Função para criar o modelo de regressão logística
def cria_modelo():
    # Cria o modelo de regressão logística
    modelo = LogisticRegression(penalty= hiperparametros['Penality'],
                                tol= float(hiperparametros['Tol']),
                                solver= hiperparametros['Solver'],
                                max_iter= hiperparametros['Max_Iteration'])
    return modelo


# Função para treinar o modelo
def treina_modelo(modelo, Xtreino, Ytreino):
    modelo.fit(Xtreino, Ytreino)


# Função para fazer previsões
def faz_previsao(modelo, Xteste):
    # Fazendo previsões no conjunto de teste
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
    st.write(f'Precisão: {Precisao:.2f}')
    st.write(f'Recall: {Recall:.2f}')
    st.write(f'F1-Score: {F1:.2f}')
    st.write(f'Acurácia: {Acuracia:.2f}')
    st.subheader('Matriz de Confusão:')
    st.write(Matriz_confusao)
    
# CHAMADA DE FUNÇÕES

# Título
titulo_app()

# Programando a Barra Lateral de Navegação da Aplicação Web
# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Configure o Modelo de ML**""")

divisao = st.sidebar.select_slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 80/20):', (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9), 0.8)

Solver = st.sidebar.selectbox('Algoritmo (padrão = lbfgs)', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.sidebar.radio("Regularização (padrão = l2):", ('none', 'l1', 'l2', 'elasticnet'), 2)
Max_Iteration = st.sidebar.select_slider("Número de Iterações (padrão = 100):", (50, 100, 500, 700, 1000), 100)
Tol = st.sidebar.text_input("Tolerância Para Critério de Parada (1e-4 a 1e-6):", "1e-4")

# Dicionário Para os Hiperparâmetros
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
hiperparametros = { 'Penality':Penality, 'Tol':Tol, 'Max_Iteration':Max_Iteration, 'Solver':Solver }

# Exibir os últimos 5 dados do Dataset
base_dados = carrega_dataset()
# pre_processar_dados(base_dados)
exibe_dataset(base_dados)
info_dataset()

# PREPARAÇÃO DOS DADOS

base_dados = remove_tempo(base_dados)

st.subheader('Divisão dos Conjuntos de Dados')
# Separando as transações em fraude e não fraude
nao_fraudes, fraudes = separa_fraudes(base_dados)
validacao, dados, validacao_real = amostra_validacao(nao_fraudes, fraudes, base_dados)

X_reamostrado, y_reamostrado = prepara_dados(base_dados)
x_treino, x_teste, y_treino, y_teste = separa_dados(X_reamostrado, y_reamostrado, divisao)

# Programando o Botão de Ação
if(st.sidebar.button("Clique Para Treinar o Modelo de Random Forest Classifier")):
    
    # Barra de progressão
    with st.spinner('Separando os Dados...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Dados de Treino e Teste Prontos!")
    
    # Cria o modelo
    st.subheader('Treinamento do Modelo')
    linear_regression = cria_modelo()
    
    # Treina o modelo
    treina_modelo(linear_regression, x_treino, y_treino)
    
    # Barra de progressão
    barra_progressao = st.progress(0)
    
    # Mostra a barra de progressão com percentual de conclusão
    for porcentagem_completada in range(100):
        time.sleep(0.1)
        barra_progressao.progress(porcentagem_completada + 1)
        
    # Info para o usuário
    with st.spinner('Treinando o Modelo...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Modelo Treinado!")
    previsao = faz_previsao(linear_regression, x_teste)
    
    # Avaliação do modelo
    st.subheader('Avaliação do Modelo')
    precisao, recall, f1, acuracia, matriz_confusao = calcula_metricas(y_teste, previsao)
    exibe_metricas(precisao, recall, f1, acuracia, matriz_confusao)

    # Teste com o modelo
    st.subheader('Validação do Modelo')
    predict = linear_regression.predict(validacao)
    dados = pd.DataFrame({'real':validacao_real, 'previsao':predict})
    st.dataframe(dados)