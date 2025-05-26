import streamlit as st
import reveal_slides as rs

st.title("Презентация проекта")

presentation_markdown = """
# Прогнозирование отказов оборудования

## Введение
- **Задача**: Предсказать отказ оборудования (Machine failure = 1) или его отсутствие (0).
- **Датасет**: AI4I 2020 Predictive Maintenance Dataset (UCI Repository).
- **Актуальность**: Снижение простоев и затрат в промышленности.

## Описание датасета
- **Источник**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601).
- **Признаки**: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
- **Цель**: Machine failure (0 или 1).

## Этапы работы
1. **Загрузка данных**: Через CSV или ucimlrepo.
2. **Предобработка**: Удаление UDI, Product ID, типов отказов; кодирование Type; масштабирование.
3. **Обучение**: Logistic Regression, Random Forest, XGBoost.
4. **Оценка**: Accuracy, Confusion Matrix, ROC-AUC.
5. **Приложение**: Streamlit с анализом и презентацией.

## Streamlit-приложение
- **Анализ и модель**: Загрузка, обучение, визуализация, предсказания.
- **Презентация**: Описание проекта в слайдах.

## Заключение
- **Результат**: Модель предсказывает отказы с высокой точностью.
- **Улучшения**: Тюнинг гиперпараметров, дополнительные модели.
"""

rs.slides(presentation_markdown, height=500, theme="serif", config={"transition": "slide"})