import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

from firebase import get_db

st.set_page_config(
    page_title="Repositório",
    layout='wide'#'centered' #'wide'
)

# st.header("Arquivos disponíveis")
header = st.container()
toolbar = st.columns([1,2,1])

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
if delete_btn:
    for k, blob in enumerate(blobs):
        if selected[k]:
            blob.delete()
            st.rerun()

if compare_btn:
    d1, d2 = mainbody2.columns([1,1])
    dd1, dd2 = mainbody3.columns([1,1])
    selected_blobs = [blobs[k] for k, x in enumerate(selected) if x]
    
    blob0 = [x for x in selected_blobs if x.name.endswith(new_btn)][0]
    data0 = BytesIO(blob0.download_as_string())
    df0 = pd.read_excel(data0)

    d2.write(blob0.name.split('/')[-1])
    dd2.dataframe(df0, hide_index=True, use_container_width=True)

    dftotal = pd.DataFrame()
    names = []
    for x in selected_blobs:
        if x.name.endswith(new_btn):
            continue
        names.append(x.name.split('/')[-1])
        datax = BytesIO(x.download_as_string())
        dfx = pd.read_excel(datax)
        dftotal = pd.concat([dftotal, dfx], ignore_index=True)
    
    d1.write(', '.join(names))
    dd1.dataframe(dftotal, hide_index=True, use_container_width=True)

