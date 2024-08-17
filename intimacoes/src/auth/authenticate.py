import jwt

import streamlit as st
from datetime import datetime, timedelta
import extra_streamlit_components as stx
from streamlit_modal import Modal

from .hasher import Hasher
from .validator import Validator
from .utils import generate_random_pw

from .exceptions import CredentialsError, ForgotError, RegisterError, ResetError, UpdateError

# Authentication GCP API
# from firebase_admin import auth
<<<<<<< HEAD:intimacoes/src/auth/authenticate.py
from src.firebase import get_db
=======
from intimacoes.src.firebase import get_db
>>>>>>> 6ac798d58cb8893d62601410105c6c9673b837e8:intimacoes/intimacoes/src/auth/authenticate.py

import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

class Authenticate:
    """
    This class will create login, logout, register user, reset password, forgot password, 
    forgot username, and modify user details widgets.
    """
    def __init__(self, cookie_name: str, key: str, cookie_expiry_days: float=0.002, 
        preauthorized: list=None, validator: Validator=None):
        """
        Create a new instance of "Authenticate".

        Parameters
        ----------
        credentials: dict
            The dictionary of usernames, names, passwords, and emails.
        cookie_name: str
            The name of the JWT cookie stored on the client's browser for passwordless reauthentication.
        key: str
            The key to be used for hashing the signature of the JWT cookie.
        cookie_expiry_days: float
            The number of days before the cookie expires on the client's browser.
        preauthorized: list
            The list of emails of unregistered users authorized to register.
        validator: Validator
            A Validator object that checks the validity of the username, name, and email fields.
        """
        # self.credentials = credentials
        # self.credentials['usernames'] = {key.lower(): value for key, value in credentials['usernames'].items()}
        _, _, self.auth = get_db()
        self.url_api = "https://identitytoolkit.googleapis.com/v1/accounts:%s"
        self.cookie_name = cookie_name
        self.key = key
        self.cookie_expiry_days = cookie_expiry_days
        self.preauthorized = preauthorized
        self.cookie_manager = stx.CookieManager()
        self.validator = validator if validator is not None else Validator()

        if 'name' not in st.session_state:
            st.session_state['name'] = None
        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = None
        if 'username' not in st.session_state:
            st.session_state['username'] = None
        if 'logout' not in st.session_state:
            st.session_state['logout'] = None

    def _token_encode(self) -> str:
        """
        Encodes the contents of the reauthentication cookie.

        Returns
        -------
        str
            The JWT cookie for passwordless reauthentication.
        """
        return jwt.encode({'name':st.session_state['name'],
            'username':st.session_state['username'],
            'exp_date':self.exp_date}, self.key, algorithm='HS256')

    def _token_decode(self) -> str:
        """
        Decodes the contents of the reauthentication cookie.

        Returns
        -------
        str
            The decoded JWT cookie for passwordless reauthentication.
        """
        try:
            return jwt.decode(self.token, self.key, algorithms=['HS256'])
        except:
            return False

    def _set_exp_date(self) -> str:
        """
        Creates the reauthentication cookie's expiry date.

        Returns
        -------
        str
            The JWT cookie's expiry timestamp in Unix epoch.
        """
        return (datetime.utcnow() + timedelta(days=self.cookie_expiry_days)).timestamp()

    def _check_pw(self) -> bool:
        """
        Checks the validity of the entered password.

        Returns
        -------
        bool
            The validity of the entered password by comparing it to the hashed password on disk.
        """
        request = requests.post(
            self.url_api % "signInWithPassword",
            params={
                "key": os.environ["KEY_API_GCP"]
            },
            data={
                "email": self.username,
                "password": self.password,
                "returnSecureToken": True
            }
        )
        request_json = request.json()
        request_dumps = json.dumps(request_json)

        if 'INVALID_PASSWORD' in request_dumps or 'EMAIL_NOT_FOUND' in request_dumps:
            return False

        elif 'email' in request_json and request_json['email'] == self.username:
            st.session_state["email"] = request_json["email"]
            return True

        # return bcrypt.checkpw(self.password.encode(), 
        #     self.credentials['usernames'][self.username]['password'].encode())

    def _check_cookie(self):
        """
        Checks the validity of the reauthentication cookie.
        """
        self.token = self.cookie_manager.get(self.cookie_name)
        if self.token is not None:
            self.token = self._token_decode()
            if self.token is not False:
                if not st.session_state['logout']:
                    if self.token['exp_date'] > datetime.utcnow().timestamp(): 
                    # if self.token['exp_date'] > (datetime.utcnow() + timedelta(days=29, hours=23, minutes=59)).timestamp():
                        if 'name' and 'username' in self.token:
                            st.session_state['name'] = self.token['name']
                            st.session_state['username'] = self.token['username']
                            st.session_state['authentication_status'] = True
    
    def _check_credentials(self, inplace: bool=True) -> bool:
        """
        Checks the validity of the entered credentials.

        Parameters
        ----------
        inplace: bool
            Inplace setting, True: authentication status will be stored in session state, 
            False: authentication status will be returned as bool.
        Returns
        -------
        bool
            Validity of entered credentials.
        """
        try:
            if self._check_pw():
                if inplace:
                    # st.session_state['name'] = self.credentials['usernames'][self.username]['name']
                    self.exp_date = self._set_exp_date()
                    self.token = self._token_encode()
                    self.cookie_manager.set(self.cookie_name, self.token,
                        expires_at=datetime.now() + timedelta(days=self.cookie_expiry_days))
                    st.session_state['authentication_status'] = True
                else:
                    return True
            else:
                if inplace:
                    st.session_state['authentication_status'] = False
                else:
                    return False
        except Exception as e:
            if e.args[0] == "KEY_API_GCP":
                st.toast("⚠️ Error 404, GCP_KEY")
                st.stop()
            print(e)
        

    def login(self, form_name: str, location: str='main') -> tuple:
        """
        Creates a login widget.

        Parameters
        ----------
        form_name: str
            The rendered name of the login form.
        location: str
            The location of the login form i.e. main or sidebar.
        Returns
        -------
        str
            Name of the authenticated user.
        bool
            The status of authentication, None: no credentials entered, 
            False: incorrect credentials, True: correct credentials.
        str
            Username of the authenticated user.
        """
        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = False
        if location not in ['main', 'sidebar']:
            raise ValueError("Location must be one of 'main' or 'sidebar'")
        if not st.session_state['authentication_status']:
            self._check_cookie()
            if not st.session_state['authentication_status']:
                if location == 'main':
                    login_form = st.form('Login')
                elif location == 'sidebar':
                    login_form = st.sidebar.form('Login')

                login_form.subheader(form_name)
                self.username = login_form.text_input('Username').lower()
                st.session_state['username'] = self.username
                self.password = login_form.text_input('Password', type='password')

                if login_form.form_submit_button('Login'):
                    self._check_credentials()
                
                self.reset_password("Forgotpassword")

        return st.session_state['name'], st.session_state['authentication_status'], st.session_state['username']

    def logout(self, button_name: str, location: str='main', key: str=None):
        """
        Creates a logout button.

        Parameters
        ----------
        button_name: str
            The rendered name of the logout button.
        location: str
            The location of the logout button i.e. main or sidebar.
        """
        if location not in ['main', 'sidebar']:
            raise ValueError("Location must be one of 'main' or 'sidebar'")
        if location == 'main':
            if st.button(button_name, key):
                self.cookie_manager.delete(self.cookie_name)
                st.session_state['logout'] = True
                st.session_state['name'] = None
                st.session_state['username'] = None
                st.session_state['authentication_status'] = None
        elif location == 'sidebar':
            if st.sidebar.button(button_name, key):
                self.cookie_manager.delete(self.cookie_name)
                st.session_state['logout'] = True
                st.session_state['name'] = None
                st.session_state['username'] = None
                st.session_state['authentication_status'] = None

    def reset_password(self, key_modal: str) -> bool:
        """
        Creates a password reset widget.

        Parameters
        ----------
        username: str
            The username of the user to reset the password for.
        form_name: str
            The rendered name of the password reset form.
        location: str
            The location of the password reset form i.e. main or sidebar.
        Returns
        -------
        str
            The status of resetting the password.
        """

        modal = Modal("Esqueci minha senha", key=key_modal)
        open_modal = st.sidebar.button("Esqueci minha senha")
        if open_modal:
            modal.open()

        if modal.is_open():
            with modal.container():
                self.email = st.text_input("Email")
                if st.button(label="Enviar", type='secondary'):
                    response = requests.post(
                        url=self.url_api % "sendOobCode",
                        params={
                            "key": os.environ["KEY_API_GPC"]
                        },
                        data={
                            "requestType": "PASSWORD_RESET",
                            "email": self.email
                        }
                    )

                    resp_json = response.json()
                    # Caso haja algum erro
                    if "error" in resp_json:
                        resp_json_error = resp_json['error']['message']

                        # Falta de email, email invalido ou email não encontrado irá alertar o mesmo tipo para segurança
                        if "MISSING_EMAIL" in resp_json_error or "INVALID_EMAIL" in resp_json_error or "EMAIL_NOT_FOUND" in resp_json_error:
                            st.error("Email inválido", icon="⚠️")

                        # Caso haja um erro inesperado
                        else:
                            st.exception(Exception("Erro inesperado, favor contatar o desenvolvedor"))

                    # Caso de certo o envio do email para alteracao de senha
                    elif "email" in resp_json:
                        st.success("Enviado com sucesso", icon="✅️")

                    # Erro inesperado
                    else:
                        st.exception(Exception("Erro inesperado, favor contatar o desenvolvedor"))
