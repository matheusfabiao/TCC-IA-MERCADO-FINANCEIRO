import streamlit as st

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
    # Título
    st.title('Bem-Vindo à Página Inicial do TCC')
    st.subheader('TCC - Trabalho de Conclusão de Curso')
    st.write("Deploy de Modelos Preditivos de Inteligência Artificial")
    st.write("Autor: Matheus Davi de Serrano Araújo Meireles")
    st.write("Autor: Matheus Fabião da Costa Pereira")
    st.write("Orientador: Leandro Santana de Melo")


def rodape():
    # Rodapé
    st.markdown('---')
    st.write('Desenvolvido por Matheus Fabião - 05/11/2023')
    
    
def formulario():
    # # Crie um formulário
    # form = st.form(key='formulario')   
    # # Crie um campo de idade
    # idade = st.number_input("Idade:", min_value=0, max_value=120, value=22)
    # # Crie um botão de envio
    # # botao_enviar = st.form_submit_button(label='Enviar')
    # # Se o formulário for enviado
    # if form.form_submit_button('Enviar'):
    #     st.success('Os dados foram recebidos com sucesso!')
    #     # Atribua os dados do formulário à sessão
    #     st.session_state["idade"] = idade
    

    with st.form(key='formulario'):
        nome = st.text_input('Nome').capitalize()
        idade = st.number_input("Idade:", min_value=0, max_value=120, value=22)
        sexo = st.radio('Sexo', ('Masculino', 'Feminino'), 0)
        profissao = st.selectbox('Profissao', ['Não qualificado e não residente', 'Não qualificado e residente', 'Qualificado', 'Altamente qualificado'])
        moradia = st.selectbox('Moradia', ['Própria', 'Gratuita', 'Alugada'])
        poupanca = st.selectbox('Poupança', ['Não Declarar', 'Pouco', 'Bastante Rico', 'Rico', 'Moderado'])
        conta_corrente = st.selectbox('Conta Corrente', ['Pouco', 'Moderado', 'Não Declarar', 'Rico'])
        valor_do_credito = st.number_input('Valor_do_credito')
        duracao = st.number_input('Duração em meses', min_value=0)
        finalidade = st.selectbox('Finalidade', ['Rádio/TV', 'Educação', 'Móveis/Equipamentos', 'Carro', 'Negócio', 'Eletrodomésticos', 'Reparos', 'Férias/Outros'])
        
        botao_enviar = st.form_submit_button(label='Enviar')
        
        if botao_enviar:
            st.success('Os dados foram recebidos com sucesso!')
            
            # Atribua os dados do formulário à sessão
            st.session_state["nome"] = nome
            st.session_state["idade"] = idade
            st.session_state["sexo"] = sexo
            st.session_state["profissao"] = profissao
            st.session_state["moradia"] = moradia
            st.session_state["poupanca"] = poupanca
            st.session_state["conta_corrente"] = conta_corrente
            st.session_state["valor_do_credito"] = valor_do_credito
            st.session_state["duracao"] = duracao
            st.session_state["finalidade"] = finalidade

st.set_page_config(
    page_title='Trabalho de Conclusão de Curso',
    page_icon='📚'
)

st.sidebar.info('Selecione uma das aplicações acima')
titulo_app()
add_logo()


formulario()




rodape()