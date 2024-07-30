import pandas as pd
from datetime import datetime

def color_dataframe(
        dataframe: pd.DataFrame,
        name_col: str,
        color: str = "red",
        value_colored: float = -1
    ) -> pd.DataFrame:
    def apply_color(row):
        return [f"background-color: {color}" if row[name_col] == value_colored else "" for _ in row]

    # Aplica a função de estilo ao DataFrame
    return dataframe.style.apply(apply_color, axis=1)

def datetime_isoformat_tratament(
        data: str
    ) -> str:
    data_raw = data.split(".")[0]
    data_raw = " ".join(data_raw.split("T"))
    return data_raw