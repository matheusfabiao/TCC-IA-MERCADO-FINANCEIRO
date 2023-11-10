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


st.set_page_config(
    page_title='Trabalho de Conclusão de Curso',
    page_icon='📚'
)

st.sidebar.info('Selecione uma das aplicações acima')
titulo_app()
add_logo()
rodape()