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


import warnings
warnings.filterwarnings('ignore')


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://yata-apix-ad9f7b35-80ad-4665-829c-2b86020b1c6c.s3-object.locaweb.com.br/cd9cdf75398747d6b544613390fa9aee.png);
                background-repeat: no-repeat;
                background-size: 100% auto;
                margin-top: 35px;
                padding-top: 75px;
                position: relative
                background-position: center;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Main Menu";
                font-family: sans-serif;
                margin-left: 20px;
                margin-top: 20px;
                font-size: 50px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    dados = r'C:\Users\Matheus Fabiao\Desktop\TCC-IA-MERCADO-FINANCEIRO\data\risco_credito.csv'
    dados = pd.read_csv(dados)
    return dados


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


def cria_modelo():
    # Cria o modelo de RFC
    modelo = RandomForestClassifier(n_estimators= hiperparametros['N_estimators'],
                                             max_depth= hiperparametros['Max_depth'],
                                             min_samples_split= hiperparametros['Min_samples_split'],
                                             class_weight= hiperparametros['Class_weight'])    
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


def normaliza_novos_dados(Novos_dados, Normalizador):
    # Criando o DataFrame
    df = pd.DataFrame(Novos_dados)

    # Exibindo o DataFrame
    st.info(f'Exibindo dados de {nome}')
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


# def salva_modelo(nome, modelo):
#     with open(nome, 'wb') as modelo_file:
#         pickle.dump(modelo, modelo_file)
        
# def salva_normalizador(nome, Normalizador):
#     with open(nome, 'wb') as scaler_file:
#         pickle.dump(Normalizador, scaler_file)


st.set_page_config(
    page_title='Previsão do Risco de Crédito',
    page_icon='💳'
)

# Programando a Barra Superior da Aplicação Web

add_logo()

# Título
titulo_app()


# Programando a Barra Lateral de Navegação da Aplicação Web

add_logo()

# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Configure o Modelo de ML**""")

ajuda_config()

divisao = st.sidebar.select_slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 80/20):', (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9), 0.8)

n_estimators = st.sidebar.number_input('Número de Estimadores (n_estimators):', min_value=100, max_value=500, step=100, value=200)
max_depth = st.sidebar.slider("Profundidade Máxima (max_depth):", 1, 20, 10)
min_samples_split = st.sidebar.slider("Mínimo de Amostras para Divisão (min_samples_split):", 2, 10, 2)
class_weight = st.sidebar.radio("Peso das Classes:", (None, 'balanced', 'balanced_subsample'))

# Dicionário Para os Hiperparâmetros
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
hiperparametros = {'N_estimators':n_estimators, 'Max_depth':max_depth, 'Min_samples_split':min_samples_split, 'Class_weight':class_weight}

# Exibir os últimos 5 dados do Dataset
base_dados = carrega_dataset()
exibe_dataset(base_dados)
info_dataset()

dados_codificados = base_dados.copy()
dados_codificados = codifica_dados(dados_codificados)

x_treino, x_teste, y_treino, y_teste = separa_dados(dados_codificados, divisao)

normalizador = declara_normalizador()

x_treino, x_teste = normaliza_dados(x_treino, x_teste, normalizador)
    
# Programando o Botão de Ação
if(st.sidebar.button("Clique Para Treinar o Modelo de Random Forest Classifier")):
    
    # Barra de progressão
    with st.spinner('Separando os Dados...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Dados de Treino e Teste Prontos!")
    
    # Cria o modelo
    st.subheader('Treinamento do Modelo')
    random_forest_classifier = cria_modelo()
    
    # Treina o modelo
    treina_modelo(random_forest_classifier, x_treino, y_treino)
    
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
    
    previsao = faz_previsao(random_forest_classifier, x_teste)

    # Avaliação do modelo

    # Acurácia do modelo
    st.subheader('Métricas')
    precisao, recall, f1, acuracia, matriz_confusao = calcula_metricas(y_teste, previsao)
    
    exibe_metricas(precisao, recall, f1, acuracia, matriz_confusao)

    # Validação cruzada
    pontuacoes_validacao_cruzada = cross_val_score(estimator=random_forest_classifier, X=x_treino, y=y_treino, cv=10)
    st.write("Média da Validação Cruzada:", pontuacoes_validacao_cruzada.mean())
    
    # VALIDAÇÃO DO MODELO
    
    st.subheader('Validação do Modelo')
    # Definição de variáveis
    nome = st.session_state["nome"]
    idade = st.session_state["idade"]
    sexo = st.session_state["sexo"]
    profissao = st.session_state["profissao"]
    moradia = st.session_state["moradia"]
    poupanca = st.session_state["poupanca"]
    conta_corrente = st.session_state["conta_corrente"]
    valor_do_credito = st.session_state["valor_do_credito"]
    duracao = st.session_state["duracao"]
    finalidade =st.session_state["finalidade"]

    # # Exibição das informações
    # st.write('Nome igual a:', nome)
    # st.write('Idade igual a:', idade)
    # st.write('Sexo igual a:', sexo)
    # st.write('Profissão igual a:', profissao)
    # st.write('Moradia igual a:', moradia)
    # st.write('Poupança igual a:', poupanca)
    # st.write('Conta Corrente igual a:', conta_corrente)
    # st.write('Valor do Crédito igual a:', valor_do_credito)
    # st.write('Duração igual a:', duracao)
    # st.write('Finalidade igual a:', finalidade)

    # Criando um dicionário com as variáveis
    data = {
        'Age': [idade],
        'Sex': [sexo],
        'Job': [profissao],
        'Housing': [moradia],
        'Saving accounts': [poupanca],
        'Checking account': [conta_corrente],
        'Credit amount': [valor_do_credito],
        'Duration': [duracao],
        'Purpose': [finalidade]
    }

    
    x = normaliza_novos_dados(data, normalizador)
    previsao = previsao_novos_dados(random_forest_classifier, x)
    
    # Salvando o modelo
    
    
    # nome_modelo = 'modelo_rfc.pkl'
    # nome_normalizador = 'normalizador.pkl'
    
    # salva_modelo(nome_modelo, random_forest_classifier)
    # salva_normalizador(nome_normalizador, normalizador)
    # st.success('O Modelo Foi Salvo')
    
    # Obrigado
    st.write("Obrigado por usar este app do Streamlit!")