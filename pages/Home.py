import streamlit as st

def titulo_app():
    # Título
    st.title('Bem-Vindo à Página Inicial do TCC')
    st.subheader('TCC - Trabalho de Conclusão de Curso')
    st.write("Deploy de Modelos Preditivos de Inteligência Artificial")
    st.write("Autor: Matheus Davi de Serrano Araújo Meireles")
    st.write("Autor: Matheus Fabião da Costa Pereira")
    st.write("Autor: Leandro Santana de Melo")

def rodape():
    # Rodapé
    st.markdown('---')
    st.write('Desenvolvido por Matheus Fabião - 05/11/2023')


def imagem_capa():
    # Imagem de capa
    st.image(r'C:\Users\Matheus Fabiao\Desktop\projetos tcc\tcc.png', use_column_width=True)

st.set_page_config(
    page_title='Trabalho de Conclusão de Curso',
    page_icon='📚'
)

st.sidebar.info('Selecione uma das aplicações acima')
titulo_app()
rodape()