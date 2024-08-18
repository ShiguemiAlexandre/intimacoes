import streamlit as st
import numpy as np
import pandas as pd
import time
from io import BytesIO
import src.functions.similarity as similarity
from streamlit_modal import Modal
import datetime
import ast
import re

from src.firebase import get_db


def validation_none_or_dict(blob, key: str) -> str:
    return blob.metadata.get(key, "Não definido") if isinstance(blob, dict) else "Não definido"

def verify_filename(filename_with_extension: str) -> str:
    _filename, extension = filename_with_extension.rsplit(".", 1)
    cont = 1
    while storage.get_blob(DEFAULT_PATH_STORAGE_COMPARE % ".".join([_filename, extension])):
        match = re.search(r'\(\d+\)$', _filename)
        if match:
            _filename = re.sub(
                r"\(\d+\)$",
                "({})".format(cont),
                _filename
            )
        else:
            _filename_struct = _filename + " (%i)"
            _filename = _filename_struct % cont
        cont += 1
    return ".".join([_filename, extension])

def verify_exists_dataframe_compareded(list_generation: list) -> list:
    blobs = storage.list_blobs(prefix="compare")
    list_filename_compareded = []
    for blob in blobs:
        list_compared = ast.literal_eval(blob.metadata["list_generation_compare"])
        if list_generation == list_compared:
            list_filename_compareded.append(blob.name.split("/", 1)[-1])
    return list_filename_compareded

def upload_dataframe_compareded():
    df_filtred, df_raw = similarity.process(st.session_state["dftotal"])
    df_gpt = similarity.compare(df_filtred, st.session_state["df0"])

    excel_buffer = BytesIO()
    df_gpt.to_excel(excel_buffer, index=False)
    excel_bytes = excel_buffer.getvalue()
    excel_buffer.close()

    blob_storage = storage.blob(st.session_state["path_storage"])

    dict_data_file = {
        "owner": st.session_state["username"],
        "compare_date": datetime.datetime.now().isoformat(),
        "list_files_compare": names,
        "list_generation_compare": generation,
        "path_storage": st.session_state["path_storage"]
    }

    blob_storage.metadata = dict_data_file

    db.collection("compare").document(st.session_state["path_storage"].split("/", 1)[-1]).set(dict_data_file)

    blob_storage.upload_from_string(
        data=excel_bytes,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.toast("Arquivo %s salvo com sucesso!" % st.session_state["path_storage"].split("/", 1)[-1])



if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is not True:
    st.title("Favor realizar login para acessar esta página 🔐")
    st.stop()

if "df0" not in st.session_state:
    st.session_state["df0"] = None

if "dftotal" not in st.session_state:
    st.session_state["dftotal"] = None

st.set_page_config(
    page_title="Repositório",
    layout='wide'#'centered' #'wide'
)

DEFAULT_PATH_STORAGE_COMPARE = "compare/%s"
header = st.container()
toolbar = st.columns([1, 1, 1, 1, 1])

mainbody1 = st.expander('Seleção de arquivos', expanded=True)
m1, m2 = mainbody1.columns([1,1])
mainbody2 = st.container()
mainbody3 = st.container()

client_folder = 'CBS'
db, storage, _ = get_db()

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
modal_filename_compareded = Modal("Existe arquivos já comparado", key="asd")
list_filename_compared = []

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
    st.session_state["df0"] = pd.read_excel(data0)

    dftotal = pd.DataFrame()
    names = []
    generation = []
    
    for x in selected_blobs:
        names.append(x.name.split('/')[-1])
        generation.append(x.generation)
        if x.name.endswith(new_btn):
            continue
        datax = BytesIO(x.download_as_string())
        dfx = pd.read_excel(datax)
        st.session_state["dftotal"] = pd.concat([dftotal, dfx], ignore_index=True)

    # dftotal é todos os arquivos selecionados menos o que foi selecionado no novo arquivo
    # blob 0 é o novo arquivo

    tab = st.tabs(["Filtrado GPT", "Filtrado", "Raw"])

    file_name_after_compare = []
    for x in selected_blobs:
        file_name_after_compare.append(x.id.split("/")[-1])
    st.session_state["path_storage"] = DEFAULT_PATH_STORAGE_COMPARE % new_btn

    if storage.get_blob(st.session_state["path_storage"]):
        st.session_state["path_storage"] = DEFAULT_PATH_STORAGE_COMPARE % verify_filename(new_btn)
        st.session_state["filenames_compareded"] = verify_exists_dataframe_compareded(generation)
        modal_filename_compareded.open()

    upload_dataframe_compareded()


title_modal_details = "Detalhes do arquivo"
modal_details = Modal(
    title=title_modal_details,
    key="details_file"
)

if modal_filename_compareded.is_open():
    with modal_filename_compareded.container():
        for filename in st.session_state["filenames_compareded"]:
            st.write(filename)
        l, _, r = st.columns([1, 2, 1])

        if l.button("Cancel", "cancel"):
            modal_filename_compareded.close()

        if r.button("Continuar", "continue"):
            upload_dataframe_compareded()

if details_btn:
    modal_details.open()

if modal_details.is_open():
    with modal_details.container():
        columns = st.columns([1, 1, 1])
        for k, blob in enumerate(blobs):
            if selected[k]:
                columns[1].text_input(
                    label="Upload feito por:",
                    value=validation_none_or_dict(blob=blob, key="owner"),
                    key="owner",
                    disabled=True,
                )
                columns[1].text_input(
                    label="Arquivo feito por:",
                    value=validation_none_or_dict(blob=blob, key="upload_time"),
                    key="time-file",
                    disabled=True,
                )
                st.caption(blob.name)
                st.caption("%s bytes" % blob.size)
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