import streamlit as st
import yaml
from src.auth.authenticate import Authenticate
import yaml
from yaml.loader import SafeLoader

import logging

MODULE_LOGGER = logging.getLogger(__name__)

with open('./intimacoes/src/config.yaml') as file:
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

