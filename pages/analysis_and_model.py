import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Анализ данных и модель")

# Data Upload
uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Первые 5 строк данных:", data.head())

    # Preprocessing
    try:
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])
        scaler = StandardScaler()
        num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        data[num_features] = scaler.fit_transform(data[num_features])
        
        # Data Splitting
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training and Evaluation
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        best_model = None
        best_roc_auc = 0
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            st.write(f"### {name}")
            st.write(f"Точность (Accuracy): {accuracy:.2f}")
            st.write("Матрица ошибок:")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            st.write("Отчет по классификации:")
            st.text(class_report)
            st.write(f"ROC-AUC: {roc_auc:.2f}")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC-кривая')
            ax.legend()
            st.pyplot(fig)

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model
        
        # Prediction Interface
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            type_input = st.selectbox("Тип (Type)", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха [K]", min_value=0.0)
            process_temp = st.number_input("Температура процесса [K]", min_value=0.0)
            rotational_speed = st.number_input("Скорость вращения [rpm]", min_value=0.0)
            torque = st.number_input("Крутящий момент [Nm]", min_value=0.0)
            tool_wear = st.number_input("Износ инструмента [min]", min_value=0.0)
            submit = st.form_submit_button("Предсказать")
            if submit:
                input_data = pd.DataFrame({
                    'Type': [le.transform([type_input])[0]],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                input_data[num_features] = scaler.transform(input_data[num_features])
                pred = best_model.predict(input_data)[0]
                pred_proba = best_model.predict_proba(input_data)[0, 1]
                st.write(f"Предсказание: {'Отказ' if pred == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {pred_proba:.2f}")
    except Exception as e:
        st.error(f"Ошибка обработки данных: {e}")
else:
    st.info("Пожалуйста, загрузите CSV-файл с данными.")