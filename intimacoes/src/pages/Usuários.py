from src.firebase import get_db
import streamlit as st
import time
import secrets

if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is not True:
    st.title("Favor realizar login para acessar esta página 🔐")
    st.stop()

def delet_user(id: str, email: str):
    auth.delete_user(id)

    time.sleep(1)
    st.toast("Email {} deletado com sucesso".format(email), icon="✅")
    time.sleep(1)
    st.rerun()
        

_, _, auth = get_db()
create, list_users = st.tabs(["🆕 Criação de usuários", "👥 Lista de usuários"])

with create:
    name = st.text_input("Nome completo do usuáro")
    email = st.text_input("Email")

    if st.button("Criar usuário"):
        with st.spinner("Criando..."):
            auth.create_user(
                email=email,
                display_name=name,
                password=secrets.token_hex(64),
                disabled=False
            )
            time.sleep(2)
            st.toast("Usuário criado com sucesso", icon="✅")

with list_users:
    for user in auth.list_users().iterate_all():
        column = st.columns([1, 1])
        column[0].text_input(label="", value=user.email, disabled=True, label_visibility="collapsed")
        bt_delet_user = column[1].button("Deletar", key=user.uid, type="primary")
        if bt_delet_user:
            delet_user(email=user.email, id=user.uid)