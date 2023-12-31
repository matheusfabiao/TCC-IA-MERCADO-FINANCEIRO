# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')

# Importando bibliotecas necessárias
import time
import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc

# Função para exibir o título do aplicativo
def titulo_app():
    # st.write("TCC - Trabalho de Conclusão de Curso")
    # st.write("Deploy de Modelo Preditivo de Machine Learning")
    # st.write("Previsão do Preço de Fechamento das Ações")
    # st.write("Autor: Matheus Fabião da Costa Pereira")
    # st.write("Autor: Matheus Davi de Serrano Araújo Meireles")
    # st.write("Orientador: Leandro Santana de Melo")
    st.title("Logistic Regression Model")
    

# Função para carregar o dataset
@st.cache_data
def carrega_dataset():
    try:
        caminho_dados = './data/creditcard.csv'
        base_dados = pd.read_csv(caminho_dados)
        dados = base_dados.copy()
        return dados
    except Exception as e:
        st.error(f'Erro ao carregar dataset: {str(e)}')
        return None


# Função para exibir o dataset
def exibe_dataset(dados) -> None:
    # Exibir os últimos 5 dados do Dataset
    st.subheader('Dataset: Detecção de Fraudes')
    st.dataframe(dados.head())


# Função para exibir informações sobre o dataset
def info_dataset():
    # Exibe um expander ao clicar no botão de ajuda
    with st.expander("ℹ️ **Informações do Dataset**"):
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
    # Exibindo o total de dados
    st.write('O número total de dados é:', len(dados))
    
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
def cria_modelo(Hiperparametros):
    # Cria o modelo de regressão logística
    modelo = LogisticRegression(penalty= Hiperparametros['Penality'],
                                tol= float(Hiperparametros['Tol']),
                                solver= Hiperparametros['Solver'],
                                max_iter= Hiperparametros['Max_Iteration'])
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
                    **Divisão dos Dados:** dividir o total de dados em x% para treino da IA. Recomendado entre 0.7 e 0.8

                    **Algoritmo Solver:** Algoritmo de otimização usado para otimizar a função de custo (e.g., "lbfgs", "liblinear", "sag", "saga") em modelos de regressão logística.
 
                    **Penalty:** Termo de penalidade aplicado à regularização (e.g., "l1" para L1 regularization, "l2" para L2 regularization) em modelos de regressão logística.
                    Recomendado: l2 para evitar *overfitting*.
 
                    **tol:** Tolerância para critério de parada, representando a precisão desejada na otimização em modelos de regressão logística.
 
                    **max_iteration:** Número máximo de iterações durante a otimização em modelos de regressão logística.
                    ''')