import streamlit as st
from utils.Global import *
from utils.ImportHome import *

st.set_page_config(
    page_title='Trabalho de Conclusão de Curso',
    page_icon='📚'
)

st.sidebar.info('Selecione uma das aplicações acima')

titulo_app()
add_logo()
formulario()
rodape()