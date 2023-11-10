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