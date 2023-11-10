# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')

# Importando bibliotecas necessárias
import streamlit as st
from utils.ImportDeteccaoFraudes import *
from utils.Global import *

# Código de Detecção de Fraudes em Transações de Cartão de Crédito
# Autor: Matheus Fabião da Costa Pereira
# Data de Criação: 30/10/2023

    
# CHAMADA DE FUNÇÕES

st.set_page_config(
    page_title='Detecção de Fraudes Bancárias',
    page_icon='💰'
)

# Título
titulo_app()

# Programando a Barra Lateral de Navegação da Aplicação Web

add_logo()

# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Configure o Modelo de ML**""")

ajuda_config()

divisao = st.sidebar.select_slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 80/20):', (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9), 0.8)

Solver = st.sidebar.selectbox('Algoritmo (padrão = lbfgs)', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.sidebar.radio("Regularização (padrão = l2):", ('none', 'l1', 'l2', 'elasticnet'), 2)
Tol = st.sidebar.selectbox('''Tolerância Para Critério de Parada                               
                                (padrão = 1e-4):''', ('1e-4', '1e-5', '1e-6'))
Max_Iteration = st.sidebar.select_slider("Número de Iterações (padrão = 100):", (50, 100, 500, 700, 1000), 100)


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
if(st.sidebar.button("Clique Para Treinar o Modelo de Logistic Regression")):
    
    # Barra de progressão
    with st.spinner('Separando os Dados...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Dados de Treino e Teste Prontos!")
    
    # Cria o modelo
    st.subheader('Treinamento do Modelo')
    linear_regression = cria_modelo(hiperparametros)
    
    try:
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
    except Exception as e:
        st.error(f"Erro durante o treinamento do modelo: {str(e)}")
        
    try:    
        previsao = faz_previsao(linear_regression, x_teste)
    
        # Avaliação do modelo
        st.subheader('Avaliação do Modelo')
        precisao, recall, f1, acuracia, matriz_confusao = calcula_metricas(y_teste, previsao)
        exibe_metricas(precisao, recall, f1, acuracia, matriz_confusao)
    except Exception as e:
        st.error(f"Erro durante o cálculo de métricas ou previsões: {str(e)}")
        
    # Teste com o modelo
    st.subheader('Validação do Modelo')
    predict = linear_regression.predict(validacao)
    dados = pd.DataFrame({'real':validacao_real, 'previsao':predict})
    st.dataframe(dados)