import streamlit as st
import numpy as np
import pandas as pd
import time
from io import BytesIO
import intimacoes.src.functions.similarity as similarity
from streamlit_modal import Modal
from intimacoes.src.functions.tratament_data import datetime_isoformat_tratament

from intimacoes.src.functions.tratament_data import color_dataframe

from intimacoes.src.firebase import get_db

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

if st.session_state["authentication_status"] != True:
    st.title("Favor realizar login para acessar esta página 🔐")
    st.stop()

st.set_page_config(
    page_title="Repositório",
    layout='wide'#'centered' #'wide'
)

# st.header("Arquivos disponíveis")
header = st.container()
toolbar = st.columns([1, 1, 1, 1, 1])

mainbody1 = st.expander('Seleção de arquivos', expanded=True)
m1, m2 = mainbody1.columns([1,1])
mainbody2 = st.container()
mainbody3 = st.container()

client_folder = 'CBS'
_, storage, _ = get_db()

########################################################################
blobs = [x for x in storage.list_blobs(prefix=f"{client_folder}")]
selected = [False for x in blobs]

m1.write("Arquivos armazenados")
for k, blob in enumerate(blobs):
    bname = blob.name.split('/')[-1]
    selected[k] = m1.checkbox(bname)
if len(blobs) == 0:
    mainbody2.warning('Nenhum arquivo encontrado')


########################################################################
compare_btn_disabled = np.sum(selected)<2
compare_btn = toolbar[0].button(
    label="Comparar",
    key="compare_button",
    disabled=compare_btn_disabled
)

modal_details = Modal(
    title="Detalhes do arquivo",
    key="details_file"
)
details_btn_disabled = np.sum(selected)==1
details_btn = toolbar[2].button(
    label="Detalhes",
    key="details_btn",
    disabled=not details_btn_disabled
)


delete_btn_disabled = np.sum(selected)==0
delete_btn = toolbar[-1].button(
    label="Apagar",
    key="delete_button",
    type="primary",
    disabled=delete_btn_disabled
)

new_btn_disabled = np.sum(selected)==0
new_btn = m2.selectbox(
    label="Novo arquivo",
    options=[blobs[k].name.split('/')[-1] for k, x in enumerate(selected) if x],
    disabled=new_btn_disabled
)

########################################################################

if compare_btn:
    d1, d2 = mainbody2.columns([1,1])
    dd1, dd2 = mainbody3.columns([1,1])
    selected_blobs = [blobs[k] for k, x in enumerate(selected) if x]
    
    # names_tabs = ["Filtrado"]
    # df_tabs = [""]
    # for x in selected_blobs:
    #     names_tabs.append(x.name)
    #     df_tabs.append(pd.read_excel(BytesIO(x.download_as_string())))
    # tabs = st.tabs(names_tabs)
    # for element, df in zip(tabs, df_tabs):
    #     element.write(df)

    blob0 = [x for x in selected_blobs if x.name.endswith(new_btn)][0]
    data0 = BytesIO(blob0.download_as_string())
    df0 = pd.read_excel(data0)

    # d2.write(blob0.name.split('/')[-1])
    # dd2.dataframe(df0, hide_index=True, use_container_width=True)

    dftotal = pd.DataFrame()
    names = []
    for x in selected_blobs:
        if x.name.endswith(new_btn):
            continue
        names.append(x.name.split('/')[-1])
        datax = BytesIO(x.download_as_string())
        dfx = pd.read_excel(datax)
        dftotal = pd.concat([dftotal, dfx], ignore_index=True)

    # dftotal é todos os arquivos selecionados menos o que foi selecionado no novo arquivo
    # blob 0 é o novo arquivo

    tab = st.tabs(["Filtrado GPT", "Filtrado", "Raw"])
    df_filtred, df_raw = similarity.process(dftotal)
    df_gpt = similarity.compare(df_filtred, df0)
    
    tab[1].dataframe(color_dataframe(df_filtred, "SIMILAR"))
    tab[2].dataframe(color_dataframe(df_raw, "SIMILAR"))
    tab[0].dataframe(color_dataframe(df_gpt, "GPTSIMILAR"))


    # d1.write(', '.join(names))
    # dd1.dataframe(dftotal, hide_index=True, use_container_width=True)

    # df_filtred = similarity.compare()
    # df_filtred_1, yyy = similarity.process(df0)
    # df_filtred_2, y = similarity.process(dftotal)
    
    # tab1, tab2 = st.tabs(["Default", "Teste"])

    # tab1.write(yyy)
    # tab2.write(y)

if details_btn:
    modal_details.open()

if modal_details.is_open():
    with modal_details.container():
        columns = st.columns([1, 1, 1])
        for k, blob in enumerate(blobs):
            if selected[k]:
                columns[1].text_input(
                    label="Upload feito por:",
                    value=blob.metadata["owner"],
                    key="owner",
                    disabled=True,
                )
                columns[1].text_input(
                    label="Arquivo feito por:",
                    value=datetime_isoformat_tratament(blob.metadata["upload_time"]),
                    key="time-file",
                    disabled=True,
                )
                st.divider()
                

if delete_btn:
    list_blobs_delet = []
    for k, blob in enumerate(blobs):
        if selected[k]:
            list_blobs_delet.append(blob)
            blob.delete()
    for x in list_blobs_delet:
        st.toast("Arquivo %s deletado" % x.name)
        time.sleep(1)
    st.rerun()