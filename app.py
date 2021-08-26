import streamlit as st
import torch
from main import predict_pipeline
from model import LangModel

lang2string = {
    'deu': 'german 🇩🇪!',
    'eng': 'english 🇬🇧!',
    'fra': 'frensh 🇫🇷!',
    'ita': 'italian 🇮🇹!',
    'por': 'portuguese 🇵🇹!',
    'spa': 'spanish 🇪🇸!',
}

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LangModel().to(device)
model.load_state_dict(torch.load('checkpoints/model.pth'))

st.title('LangIdentify 🏳️‍🌈')
st.write('This app identifies the language of your text.\nIt detects **_6_** different languages: 🇩🇪 🇬🇧 🇫🇷 🇮🇹 🇵🇹 🇪🇸')

text = st.text_input('Text')

st.write('This text is written in ', lang2string[predict_pipeline(model, [text])[0]] if len(text) > 0 else '?')
