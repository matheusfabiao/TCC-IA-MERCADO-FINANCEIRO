# Importando as bibliotecas necessárias
import time
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import ceil
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import date
from plotly import graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')


def titulo_app():
    # st.write("TCC - Trabalho de Conclusão de Curso")
    # st.write("Deploy de Modelo Preditivo de Machine Learning")
    # st.write("Previsão do Preço de Fechamento das Ações")
    # st.write("Autor: Matheus Fabião da Costa Pereira")
    # st.write("Autor: Matheus Davi de Serrano Araújo Meireles")
    # st.write("Orientador: Leandro Santana de Melo")
    st.title("Long Short Term Memory (LSTM)")


# Função que converte o tempo em str
def converter_tempo(Intervalo) -> str:
    Intervalo = str(Intervalo) + 'y'
    return Intervalo


# Função para carregar o dataset
@st.cache_data
def carregar_dataset(Ativo, Intervalo):
    try:
        dados = yf.download(tickers=Ativo, period=Intervalo)
        return dados
    except Exception as e:
        st.error(f'Erro ao carregar dados: {str(e)}')
        return None


def adiciona_informacoes(dados):
    # Cálculo de Retornos Diários
    dados['Daily Return'] = dados['Close'].pct_change()
    # Cálculo de Médias Móveis Simples (SMA)
    dados['SMA_50'] = dados['Close'].rolling(window=50).mean().round(2)
    dados['SMA_200'] = dados['Close'].rolling(window=200).mean().round(2)
    # Cálculo de Volatilidade
    rolling_volatility = dados['Close'].rolling(window=252).std()  # 252 trading days in a year    
    return dados, rolling_volatility


def exibe_dataset(dados) -> None:
    # Exibir os últimos 5 dados do Dataset   
    st.subheader(f'Dataset: {ativo}')
    st.dataframe(dados)
    

def info_dataset():
    # Exibe um expander ao clicar no botão de ajuda
    with st.expander("ℹ️ **Informações do Dataset**"):
        st.write('''
         **Date:** Data da observação das informações financeiras.                  
         **Open:** Preço de abertura do ativo no início do dia.                  
         **High:** Preço mais alto atingido durante o dia.                  
         **Low:** Preço mais baixo atingido durante o dia.                  
         **Close:** Preço de fechamento do ativo no final do dia.                  
         **Adj Close:** Preço de fechamento ajustado para eventos como dividendos.                  
         **Volume:** Volume de negociações do ativo durante o dia.                  
         **Daily Return:** Representa a variação percentual diária do preço de fechamento do ativo.                  
         **SMA_50:** Média Móvel Simples (SMA) de 50 dias do preço de fechamento, usada para suavizar tendências de curto prazo.                  
         **SMA_200:** Média Móvel Simples (SMA) de 200 dias do preço de fechamento, usada para suavizar tendências de longo prazo.
         ''')


def plot_dataset(dados, rolling_volatility):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Open'], name='Open'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['High'], name='High'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Low'], name='Low'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Adj Close'], name='Adj Close'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Volume'], name='Volume'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Daily Return'], name='Daily Return'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_50'], name='SMA 50'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_200'], name='SMA 200'))
    fig.add_trace(go.Scatter(x=dados.index, y=rolling_volatility, name='Volatility'))
    
    fig.layout.update(title_text=f'Histórico do Ativo: {ativo}', xaxis_rangeslider_visible=True)
    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(title_text='Valor')
    st.plotly_chart(fig)


def plot_dataset_fechamento(dados, Ativo):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Close'))
    
    fig.layout.update(title_text=f'Histórico de Fechamento: {Ativo}', xaxis_rangeslider_visible=True)
    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(title_text='Preço')
    st.plotly_chart(fig)


def precos_fechamento(dados):
    preco_maximo_historico = dados['Close'].max()
    preco_minimo_historico = dados['Close'].min()
    st.write('Maior Preço de Fechamento Registrado:', preco_maximo_historico)
    st.write('Menor Preço de Fechamento Registrado:', preco_minimo_historico)
    return preco_maximo_historico, preco_minimo_historico


def filtra_dados(dados):
    # Filtrando apenas os dados de fechamento do ativo
    cotacoes_df = dados.filter(['Close'])
    cotacoes = cotacoes_df.values
    return cotacoes_df, cotacoes


def normaliza_dados(dados, Cotacoes):
    # Normalizando os dados entre 0 e 1
    normalizador = MinMaxScaler(feature_range=(0, 1))
    cotacoes_normalizadas = normalizador.fit_transform(Cotacoes)
    return cotacoes_normalizadas, normalizador


def divide_dados_treino(split, Cotacoes, Cotacoes_normalizadas):
    # Determinando o número de dias para treinamento
    dias_treinamento = ceil(len(Cotacoes) * split)
    cotacoes_treinamento = Cotacoes_normalizadas[:dias_treinamento]
    return dias_treinamento, cotacoes_treinamento


def prepara_dados_treino(Cotacoes_treinamento, Previsao_dias):
    # Preparando dados para treinamento da rede neural
    x_treino = []
    y_treino = []

    # Criando sequências de dados para x_treino e y_treino
    for i in range(Previsao_dias, len(Cotacoes_treinamento)):
        x_treino.append(Cotacoes_treinamento[i-Previsao_dias:i])
        y_treino.append(Cotacoes_treinamento[i])

    # Convertendo listas em arrays numpy
    x_treino, y_treino = np.array(x_treino), np.array(y_treino)

    # Redimensionando o array de entrada para o modelo LSTM
    x_treino = np.reshape(x_treino, (len(x_treino), Previsao_dias, 1))
    return x_treino, y_treino


def divide_dados_teste(Dias_treinamento, Cotacoes_normalizadas, Previsao_dias):
    # Preparando dados de teste
    cotacoes_teste = Cotacoes_normalizadas[Dias_treinamento - Previsao_dias:]
    return cotacoes_teste


def prepara_dados_teste(Dias_treinamento, Previsao_dias, Cotacoes_teste):
    x_teste = []
    y_teste = cotacoes[Dias_treinamento:]

    # Criando sequências de dados para x_teste
    for i in range(Previsao_dias, len(Cotacoes_teste)):
        x_teste.append(Cotacoes_teste[i-Previsao_dias:i])

    # Convertendo x_teste em um array numpy e redimensionando
    x_teste = np.array(x_teste)
    x_teste = np.reshape(x_teste, (len(x_teste), Previsao_dias, 1))
    return x_teste, y_teste


# Função para plotar o gráfico de divisão
def plotar_divisao(Ativo, Cotacoes_df, Dias_treinamento):
    # Definindo a cor de fundo
    plt.style.use('dark_background')
    
    # Criando a figura
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Deixar o fundo transparente
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Definindo o título do gráfico
    ax.set_title('Divisão dos Dados para Treino e Teste: ' + Ativo)

    # Plotando a quantidade de dados de treino e teste
    ax.plot(Cotacoes_df.index[:Dias_treinamento], Cotacoes_df[:Dias_treinamento], color='orange', label='Treinamento')
    ax.plot(Cotacoes_df.index[Dias_treinamento:], Cotacoes_df[Dias_treinamento:], color='green', label='Testes')

    # Configurando rótulos dos eixos
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')

    # Definindo a legenda do gráfico
    ax.legend(loc='upper left')

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)


def cria_modelo():
    # Criando um modelo sequencial de rede neural
    modelo = Sequential()
    return modelo


def adiciona_camadas(modelo, X_treino, Taxa_dropout):
    # Adicionando camadas LSTM
    modelo.add(LSTM(units=50, return_sequences=True, input_shape=(X_treino.shape[1], 1)))
    modelo.add(Dropout(Taxa_dropout))
    modelo.add(LSTM(units=50, return_sequences=False))
    modelo.add(Dropout(Taxa_dropout))

    # Adicionando camadas densas (fully connected)
    modelo.add(Dense(units=25))
    modelo.add(Dense(units=1))  # Previsão do próximo preço de fechamento
    return modelo


def compila_modelo(modelo):
    # Compilando o modelo
    modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])


def treina_modelo(modelo, X_treino, Y_treino, lote, epocas):
    # Treinando o modelo
    modelo.fit(X_treino, Y_treino, batch_size=lote, epochs=epocas)


# Função para plotar o gráfico de perdas do modelo durante o treinamento
# Mostra que a perda caiu consideravelmente e o modelo treinou bem
def plotar_loss(modelo):
    # Definindo a cor de fundo
    plt.style.use('dark_background')
    
    # Criando a figura
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Deixar o fundo transparente
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Definindo o título do gráfico
    ax.set_title('Gráfico de Perda')

    # Plotando a quantidade de dados de treino e teste
    perda = modelo.history.history['loss']
    ax.plot(perda, label='Perda')
    
    st.pyplot(fig)
    st.write('Este gráfico exibe a perda do modelo durante o treinamento, mostrando que ela caiu consideravelmente e o modelo treinou bem.')


def faz_previsao(modelo, X_teste, Normalizador):
    previsoes = modelo.predict(X_teste)
    previsoes = Normalizador.inverse_transform(previsoes)
    return previsoes


def calcula_metricas(Previsoes, Y_teste):
    # Calcular o Erro Médio Absoluto (MAE)
    mae = mean_absolute_error(Y_teste, Previsoes)
    # Calcular o Erro Médio Quadrático (MSE)
    mse = mean_squared_error(Y_teste, Previsoes)
    # Calcular o Raiz do Erro Médio Quadrático (RMSE)
    rmse = np.sqrt(mse)
    # Calcular o Erro Absoluto Percentual Médio (MAPE)
    mape = np.mean(np.abs((Y_teste - Previsoes) / Y_teste)) * 100
    # Calcular o Coeficiente de Determinação (R²)
    r2 = r2_score(Y_teste, Previsoes)

    return mae, mse, rmse, mape, r2

def exibe_metricas(mae_, mse_, rmse_, mape_, r2_):
    # Exibir as métricas na tela
    st.subheader('Métricas:')
    st.write('Erro Médio Absoluto (MAE):', mae_)
    st.write('Erro Médio Quadrático (MSE):', mse_)
    st.write('Raiz do Erro Médio Quadrático (RMSE):', rmse_)
    st.write('Erro Absoluto Percentual Médio (MAPE):', mape_)
    st.write('Coeficiente de Determinação (R²):', r2_)


def dados_validacao(Cotacoes_df, Dias_treinamento, Previsoes):
    # Separando dados de treinamento e validação
    treino = Cotacoes_df[:Dias_treinamento]
    valido = Cotacoes_df[Dias_treinamento:]
    
    # Adicionando as previsões ao DataFrame de dados de validação
    valido['Previsões'] = Previsoes
    return treino, valido


# Função para plotar os dados de treinamento, dados reais e previsões
def plotar_dados_e_previsoes(Ativo, Treino, Valido):
    trace_historico = go.Scatter(x=Treino.index, y=Treino['Close'], mode='lines', name='Histórico')
    trace_valor_real = go.Scatter(x=Valido.index, y=Valido['Close'], mode='lines', name='Valor real', line=dict(color='red'))
    trace_previsao = go.Scatter(x=Valido.index, y=Valido['Previsões'], mode='lines', name='Previsão', line=dict(color='orange'))

    layout = go.Layout(
        title='Ativo ' + Ativo,
        xaxis=dict(title='Data'),
        yaxis=dict(title='Preço de fechamento (R$)'),
        showlegend=True
    )

    # Adicionar rangeslider visível
    layout.update(xaxis_rangeslider_visible=True)

    fig = go.Figure(data=[trace_historico, trace_valor_real, trace_previsao], layout=layout)

    st.plotly_chart(fig)


def previsao_dia_seguinte(Cotacoes_normalizadas, Previsao_dias, modelo, Normalizador):
    # Coletando os dados mais recentes para previsão (últimos 100 pontos)
    dados_recentes = Cotacoes_normalizadas[-Previsao_dias:]

    # Redimensionando os dados para o formato esperado pelo modelo LSTM
    dados_recentes = np.array([dados_recentes])
    dados_recentes = np.reshape(dados_recentes, (dados_recentes.shape[0], Previsao_dias, 1))

    # Fazendo a previsão para o próximo dia
    previsao_proximo_dia = modelo.predict(dados_recentes)

    # Invertendo a normalização para obter o valor em reais
    previsao_proximo_dia = Normalizador.inverse_transform(previsao_proximo_dia)

    # Exibindo a previsão para o próximo dia
    st.write('Previsão de preço de fechamento para o próximo dia:', previsao_proximo_dia[0][0])
    
    
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


st.set_page_config(
    page_title='Previsão da Bolsa de Valores',
    page_icon='📈'
)

# Programando a Barra Superior da Aplicação Web
# Título
titulo_app()

# Programando a Barra Lateral de Navegação da Aplicação Web
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
exibe_dataset(dados_do_ativo)
info_dataset()
st.subheader('Análise do Ativo', ativo)
plot_dataset(dados_do_ativo, volatilidade)
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
x_teste, y_teste = prepara_dados_teste(dias_treinamento, previsao_dias, cotacoes_teste)
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
    adiciona_camadas(lstm, x_treino)
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
        mae, mse, rmse, mape, r2 = calcula_metricas(previsoes, y_teste)
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