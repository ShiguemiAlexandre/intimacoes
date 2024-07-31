# -*- coding: utf-8 -*-
import streamlit as st
import socket
import os
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import (  # noqa: F401
    credentials,
    auth,
    firestore,
    storage,
)
from firebase_admin._auth_utils import (
    UserNotFoundError
)

load_dotenv()

def is_gce_instance():
    """
    Check if it's GCE instance via DNS lookup to metadata server.
    """
    try:
        socket.getaddrinfo('metadata.google.internal', 80)
    except socket.gaierror:
        return False
    return True



@st.cache_resource
def get_db():
    try:
        app = firebase_admin.get_app()
    except ValueError as e:
        fbpath = os.getenv('GLOBAL_PATH_FIREBASE_JSON')
        cred = credentials.Certificate(fbpath)
        firebase_admin.initialize_app(cred)

    firebase_db = firestore.client()
    firebase_storage = storage.bucket(os.getenv('BUCKET_NAME'))

    return firebase_db, firebase_storage, auth

    # if is_gce_instance():
    #     firebase_app = firebase_admin.initialize_app()
    # else:
    #     fbpath = os.getenv('GLOBAL_PATH_FIREBASE_JSON')
    #     cred = credentials.Certificate(fbpath)
    #     firebase_app = firebase_admin.initialize_app(cred)

    # firebase_db = firestore.client()
    # firebase_storage = storage.bucket(os.getenv('BUCKET_NAME'))

    # return firebase_db, firebase_storage, auth

FIRESTORE_TIMESTAMP = firestore.firestore.SERVER_TIMESTAMP
FIRESTORE_DELETE_FIELD = firestore.firestore.DELETE_FIELD
FIRESTORE_INCREMENT = firestore.firestore.Increment

