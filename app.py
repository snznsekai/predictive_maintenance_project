import streamlit as st
from presentation import presentation_page

def analysis_and_model_page():
    # Импорт функции из analysis_and_model.py
    from analysis_and_model import analysis_and_model_page
    analysis_and_model_page()

def presentation_page():
    # Импорт функции из presentation.py
    from presentation import presentation_page
    presentation_page()

pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page,
}

selected_page = st.sidebar.radio("Навигация", list(pages.keys()))
pages[selected_page]()