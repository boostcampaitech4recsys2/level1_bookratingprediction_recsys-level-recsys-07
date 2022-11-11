import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="Book Recommender", layout="wide")
model = torch.load('./models/NCF_model.pt')
model.eval()

def set_value(key):
    st.session_state[key] = st.session_state["key_" + key]
    
def set_status(status):
    st.session_state["status"] = status

STATE_KEYS_VALS = [
    ("get_user_age", 30),
    ("user_city","ottawa"),
    ("status", False),
    ("clicked", False),
    ("top_k", 10),
]
for k, v in STATE_KEYS_VALS:
    if k not in st.session_state:
        st.session_state[k] = v

#############################################################
###                  Define Side-bar View                 ###
#############################################################
st.sidebar.title("정보를 입력해주세요")

# 나이 정보 받기
user_age = st.sidebar.selectbox("나이를 선택하세요.",
                          list(range(1,150)),
                          index=29)

# 지역 정보 받기
location = {"ottawa":['ontario','canada'],
            "seattle":['washington','usa'],
            "seoul":['seoul','southkorea'],
            "sanfrancisco":['california','usa']}
user_city = st.sidebar.selectbox("도시를 선택하세요.",
                          ["ottawa",
                           "seattle",
                           "seoul",
                           "sanfrancisco",])
user_state = location[user_city][0]
user_country = location[user_city][1]

# top k 받기
st.sidebar.number_input(
    "How many books do you want to get recommended?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["top_k"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("top_k",),
    key="key_top_k",
)

st.sidebar.button(
    "책 추천 시작!",
    on_click=set_status,
    args=(True,),
    disabled=st.session_state["status"],
)

#############################################################
###                    Define Main View                   ###
#############################################################

st.title("Book Recomender with Streamlit by 권준혁")

# When the start button has been clicked from the side-bar
if st.session_state["status"]:
    _top_100 = pd.read_csv('top100.csv')
    _top_100.rename(columns={'Unnamed: 0':'isbn'}, inplace=True)
    