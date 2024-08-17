import streamlit as st
from src.firebase import get_db
import pandas as pd
from io import BytesIO
import ast

if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is not True:
    st.title("Favor realizar login para acessar esta página 🔐")
    st.stop()

_, storage, _ = get_db()
container_metric = st.container()
container_multiselect = st.container()

st.divider()

blobs = storage.list_blobs(prefix="compare")
list_files_compare = []
value = 0
for value, blob in enumerate(blobs):
    st.write(blob.generation)
    for name_file in ast.literal_eval(blob.metadata["list_files_compare"]):
        list_files_compare.append(name_file)
    # dataframe = pd.read_excel(BytesIO(blob.download_as_string()))
    # st.dataframe(dataframe)

files_compare_selected = container_multiselect.multiselect(
    label="Arquivos comparados",
    options=list_files_compare,
    placeholder="Selecione o(s) arquivo(s) comparado(s)"
)

container_metric.metric(
    label="Quantidade de arquivos comparados",
    value=value + 1
)
