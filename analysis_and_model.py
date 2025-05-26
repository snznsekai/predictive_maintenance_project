import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data = data.rename(columns={
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min'
})
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        
        # Масштабирование числовых признаков
        scaler = StandardScaler()
        numerical_features = [
    'Type',
    'Air_temperature_K', 
    'Process_temperature_K', 
    'Rotational_speed_rpm', 
    'Torque_Nm', 
    'Tool_wear_min'
]
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        
        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        # Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Масштабирование данных для SVM
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            if name == "SVM":
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
        
        # Оценка моделей
        st.header("Результаты оценки моделей")
        plt.figure(figsize=(10, 6))
        
        for name, model in models.items():
            # Предсказания
            if name == "SVM":
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Визуализация ROC-кривой
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            
            # Вывод результатов в Streamlit
            st.subheader(name)
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
                
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            
            st.text(f"Classification Report:\n{class_report}")
            st.markdown("---")
        
        # Общий график ROC-кривых
        st.subheader("Сравнение ROC-кривых")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые')
        plt.legend()
        st.pyplot(plt)