# import yaml
# from src.auth.authenticate import Authenticate
# import yaml
# from yaml.loader import SafeLoader
# import streamlit as st

# def auth_credentials():
#     authenticator = Authenticate(
#         cookie_name = config['cookie']['name'],
#         key = config['cookie']['key'],
#         cookie_expiry_days = config['cookie']['expiry_days'],
#         preauthorized = config['preauthorized']
#     )

#     return authenticator

# with open('./intimacoes/src/config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# auth_credentials()
# print(st.session_state())
