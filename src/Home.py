import streamlit as st
from auth.authenticate import Authenticate
import yaml
from yaml.loader import SafeLoader
from firebase import get_db

import logging

MODULE_LOGGER = logging.getLogger(__name__)

with open('./config.yaml') as file:
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

import streamlit as st
import time

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

st.button("Rerun")

authenticator.logout('Logout', 'sidebar', key='finalizar')