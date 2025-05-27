import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")
    
    # Содержание презентации в формате Markdown
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    
    ## Введение
    - Задача: Предсказание отказов оборудования (бинарная классификация)
    - Датасет: AI4I 2020 Predictive Maintenance Dataset
    - Целевая переменная: Machine failure (0/1)
    
    ---
    
    ## Этапы работы
    1. Загрузка и предобработка данных
    2. Обучение модели Random Forest
    3. Оценка метрик (Accuracy, ROC-AUC)
    4. Разработка Streamlit-приложения
    
    ---
    
    ## Технологии
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas/NumPy
    
    ---
    
    ## Результаты
    - Accuracy модели: 0.96
    - ROC-AUC: 0.98
    - Приложение доступно в GitHub
    """
    
    # Настройки презентации
    with st.sidebar:
        st.header("Настройки слайдов")
        theme = st.selectbox("Тема", ["black", "white", "league"])
        height = st.slider("Высота слайдов", 400, 800, 600)
    
    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": "slide",
            "separator": "^---$",
            "enableLinks": True
        }
    )
