import streamlit as st
import magic
from datetime import datetime
from src.firebase import get_db


if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is not True:
    st.title("Favor realizar login para acessar esta página 🔐")
    st.stop()

st.set_page_config(
    page_title='Intimações Juristecplus',
    layout='centered',
    initial_sidebar_state='expanded',
)

st.header("Upload de Arquivos")

file_upload = st.file_uploader(
    "Selecione um arquivo",
    accept_multiple_files=True,
    type=['.xlsx'],
)

CLIENT_FOLDER = 'CBS'
_, storage, _ = get_db()

if file_upload is not None:
    ready=False
    for _file in file_upload:
        fname = f'{CLIENT_FOLDER}/{_file.name}'

        # verificando se arquivo ja existe
        blob = storage.get_blob(fname)
        if blob:
            st.error(''.join((
                f'Arquivo "',
                fname.split('/')[-1],
                '" já existe, remova-o do ',
                'repositório antes de fazer upload novamente'
                )))
            st.stop()
            
        # Validando o tipo do arquivo
        data_bytes = _file.getvalue()
        ext = magic.from_buffer(
            buffer=data_bytes,
            mime=True
        )
        blob = storage.blob(fname)
        blob.metadata = {
            "owner": st.session_state["username"],
            "upload_time": datetime.now().isoformat()
        }
        blob.upload_from_string(
            data=data_bytes,
            content_type=ext
        )
        ready=True
    
    if ready:
        st.switch_page('pages/Repositório.py')
    # bytes_data = BytesIO()

    # df = pd.read_excel(bytes_data, engine='openpyxl')
    # print(df)



