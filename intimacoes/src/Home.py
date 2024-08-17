import streamlit as st
<<<<<<< HEAD:intimacoes/src/Home.py
import yaml
from src.auth.authenticate import Authenticate
=======
from intimacoes.src.auth.authenticate import Authenticate
>>>>>>> 6ac798d58cb8893d62601410105c6c9673b837e8:intimacoes/intimacoes/src/Home.py
import yaml
from yaml.loader import SafeLoader

import logging

MODULE_LOGGER = logging.getLogger(__name__)

<<<<<<< HEAD:intimacoes/src/Home.py
with open('./intimacoes/src/config.yaml') as file:
=======
with open('./intimacoes/intimacoes/config.yaml') as file:
>>>>>>> 6ac798d58cb8893d62601410105c6c9673b837e8:intimacoes/intimacoes/src/Home.py
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    cookie_name = config['cookie']['name'],
    key = config['cookie']['key'],
    cookie_expiry_days = config['cookie']['expiry_days'],
    preauthorized = config['preauthorized']
)

name, authentication_status, username = authenticator.login("Login", "sidebar")


if authentication_status == None:
    st.stop()

elif authentication_status == False:    
    st.sidebar.error("Email ou senha inválidos. Por favor, verifique suas credenciais e tente novamente.")
    st.toast("❌Email ou senha inválidos. Por favor, verifique suas credenciais e tente novamente.")
    st.stop()

authenticator.logout('Logout', 'sidebar', key='finalizar')

