# Importando as bibliotecas necessárias
import streamlit as st
from utils.ImportPrevisaoPrecoAcoes import *
from utils.Global import *

# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Previsão da Bolsa de Valores',
    page_icon='📈'
)

# Programando a Barra Superior da Aplicação Web
# Título
titulo_app()

# Programando a Barra Lateral de Navegação da Aplicação Web

add_logo()

# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Selecione o Dataset Desejado**""")

ativos = ('PETR4.SA', 'MGLU3.SA', 'ITUB4.SA', 'CIEL3.SA', 'BBDC4.SA', 'BBAS3.SA', 'LREN3.SA', 'B3SA3.SA', 'VALE3.SA', 'AAPL', 'GOOG', 'MSFT', 'TSLA', 'NFLX')

# Definindo o ativo, data de início e data de fim para a coleta de dados
ativo = st.sidebar.selectbox('Dataset', ativos)
ajuda_config()

intervalo_tempo = st.sidebar.select_slider('Escolha o Intervalo de Tempo para Treinamento (padrão = 20 anos):', (10, 15, 20, 25, 30), 20)

divisao = st.sidebar.select_slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 80/20):', (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9), 0.8)

# https://keras.io/api/layers/recurrent_layers/lstm/

taxa_dropout = st.sidebar.selectbox('Escolha o Percentual da Taxa de Dropout (Padrão = 0.2):', (0.2, 0.3, 0.4, 0.5), 0)
# taxa_dropout = st.sidebar.text_input(label='Escolha o Percentual da Taxa de Dropout (Recomendado = entre 0.2 e 0.5):', max_chars=3, value=0.2)

batch_size = st.sidebar.select_slider('Escolha o Tamanho Do Número de Amostras Para a Rede Neural (padrão = 1024 unidades):',(128, 256, 512, 1024), 1024)

# Janela temporal
previsao_dias = st.sidebar.select_slider("Escolha o Tamanho da Janela de Entrada (padrão = 15 dias)", (5, 10, 15, 30, 60), 15)

epochs = st.sidebar.radio('Escolha o número de épocas para treinamento (padrão = 100 épocas):', [50, 100, 200], 1)

# DATASET
intervalo_tempo = converter_tempo(intervalo_tempo)
dados_do_ativo = carregar_dataset(ativo, intervalo_tempo)
# dados_do_ativo = limpa_dados(dados_do_ativo)
dados_do_ativo, volatilidade = adiciona_informacoes(dados_do_ativo)
exibe_dataset(dados_do_ativo, ativo)
info_dataset()
st.subheader('Análise do Ativo', ativo)
plot_dataset(dados_do_ativo, volatilidade, ativo)
plot_dataset_fechamento(dados_do_ativo, ativo)
preco_maximo_historico, preco_minimo_historico = precos_fechamento(dados_do_ativo)

# PREPARAÇÃO DOS DADOS
st.subheader('Divisão dos Conjuntos de Dados')
cotacoes_df, cotacoes = filtra_dados(dados_do_ativo)
cotacoes_normalizadas, normalizador = normaliza_dados(dados_do_ativo, cotacoes)
# Lembrar de testar instanciar o normalizador, ao invés de colocá-lo na função
dias_treinamento, cotacoes_treinamento = divide_dados_treino(divisao, cotacoes, cotacoes_normalizadas)
    
x_treino, y_treino = prepara_dados_treino(cotacoes_treinamento, previsao_dias)
cotacoes_teste = divide_dados_teste(dias_treinamento, cotacoes_normalizadas, previsao_dias)
x_teste, y_teste = prepara_dados_teste(dias_treinamento, previsao_dias, cotacoes_teste, cotacoes)
plotar_divisao(ativo, cotacoes_df, dias_treinamento)
st.write('Tamanho Total do Conjunto de Dados:', len(dados_do_ativo))

# Programando o Botão de Ação
if(st.sidebar.button("Clique Para Treinar a Rede Neural LSTM")):
    
    # Barra de progressão
    with st.spinner('Separando os Dados...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Dados de Treino e Teste Prontos!")

    # CRIAÇÃO DO MODELO
    st.subheader('Treinamento do Modelo')
    lstm = cria_modelo()
    adiciona_camadas(lstm, x_treino, taxa_dropout)
    compila_modelo(lstm)
    # Barra de progressão
    barra_progressao = st.progress(0)
    
    try:
        treina_modelo(lstm, x_treino, y_treino, batch_size, epochs)
        
        # Mostra a barra de progressão com percentual de conclusão
        for porcentagem_completada in range(100):
            time.sleep(0.1)
            barra_progressao.progress(porcentagem_completada + 1)
            
        # Info para o usuário
        with st.spinner('Treinando o Modelo...'):
            time.sleep(3)

        # Info de sucesso
        st.success("Modelo Treinado!")
        plotar_loss(lstm)
    except Exception as e:
        st.error(f"Erro durante o treinamento do modelo: {str(e)}")
        
    try:
        previsoes = faz_previsao(lstm, x_teste, normalizador)
        # Calcula as métricas
        mae, mse, rmse, mape, r2 = calcula_metricas(y_teste, previsoes)
        # Exibe as métricas no Streamlit
        exibe_metricas(mae, mse, rmse, mape, r2)
        treino, valido = dados_validacao(cotacoes_df, dias_treinamento, previsoes)
        st.subheader('Validação do Modelo')
        plotar_dados_e_previsoes(ativo, treino, valido)
    except Exception as e:
        st.error(f"Erro durante o cálculo de métricas ou previsões: {str(e)}")

    # VALIDAÇÃO DO MODELO
    st.subheader('Previsão Para o Dia de Amanhã')
    previsao_dia_seguinte(cotacoes_normalizadas, previsao_dias, lstm, normalizador)
    
     # Obrigado
    st.write("Obrigado por usar este app do Streamlit!")