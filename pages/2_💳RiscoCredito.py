# Importando bibliotecas
import streamlit as st
# import pickle
from utils.ImportRiscoCredito import *


import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Previsão do Risco de Crédito',
    page_icon='💳'
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
    random_forest_classifier = cria_modelo(hiperparametros)
    
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

    x = normaliza_novos_dados(data, normalizador, nome)
    previsao = previsao_novos_dados(random_forest_classifier, x)
    
    # Obrigado
    st.write("Obrigado por usar este app do Streamlit!")