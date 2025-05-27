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
import numpy as np

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Initialize LabelEncoder and StandardScaler
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    # Choose input method
    input_method = st.radio("Выберите способ ввода данных:", ("Загрузка CSV", "Ручной ввод"))
    
    if input_method == "Загрузка CSV":
        # Existing CSV upload functionality
        uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Preprocessing data
            data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            data = data.rename(columns={
                'Air temperature [K]': 'Air_temperature_K',
                'Process temperature [K]': 'Process_temperature_K',
                'Rotational speed [rpm]': 'Rotational_speed_rpm',
                'Torque [Nm]': 'Torque_Nm',
                'Tool wear [min]': 'Tool_wear_min'
            })
            data['Type'] = label_encoder.fit_transform(data['Type'])
            
            # Scale numerical features
            numerical_features = [
                'Type',
                'Air_temperature_K', 
                'Process_temperature_K', 
                'Rotational_speed_rpm', 
                'Torque_Nm', 
                'Tool_wear_min'
            ]
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            
            # Split data
            X = data.drop(columns=['Machine failure'])
            y = data['Machine failure']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y
            )
            
            # Train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                "SVM": SVC(kernel='rbf', probability=True, random_state=42)
            }
            
            # Scale data for SVM
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for name, model in models.items():
                if name == "SVM":
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
            
            # Evaluate models
            st.header("Результаты оценки моделей")
            plt.figure(figsize=(10, 6))
            
            for name, model in models.items():
                # Predictions
                if name == "SVM":
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
                
                # Display results in Streamlit
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
            
            # Display combined ROC curves
            st.subheader("Сравнение ROC-кривых")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC-кривые')
            plt.legend()
            st.pyplot(plt)
            
            # Store models and scaler for manual input predictions
            st.session_state['models'] = models
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            
    else:
        # Manual input form
        st.header("Ручной ввод данных")
        with st.form("manual_input_form"):
            type_input = st.selectbox("Тип оборудования", ['L', 'M', 'H'])
            air_temp = st.number_input("Температура воздуха [K]", min_value=0.0, value=298.0)
            process_temp = st.number_input("Температура процесса [K]", min_value=0.0, value=308.0)
            rotational_speed = st.number_input("Скорость вращения [rpm]", min_value=0.0, value=1500.0)
            torque = st.number_input("Крутящий момент [Nm]", min_value=0.0, value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", min_value=0.0, value=0.0)
            submit_button = st.form_submit_button("Предсказать")
        
        if submit_button:
            # Create a DataFrame from manual input
            manual_data = pd.DataFrame({
                'Type': [type_input],
                'Air_temperature_K': [air_temp],
                'Process_temperature_K': [process_temp],
                'Rotational_speed_rpm': [rotational_speed],
                'Torque_Nm': [torque],
                'Tool_wear_min': [tool_wear]
            })
            
            # Preprocess manual input
            manual_data['Type'] = st.session_state.get('label_encoder', LabelEncoder()).transform(manual_data['Type'])
            numerical_features = [
                'Type',
                'Air_temperature_K', 
                'Process_temperature_K', 
                'Rotational_speed_rpm', 
                'Torque_Nm', 
                'Tool_wear_min'
            ]
            manual_data_scaled = st.session_state.get('scaler', StandardScaler()).transform(manual_data[numerical_features])
            
            # Make predictions
            st.header("Результаты предсказания")
            models = st.session_state.get('models', {})
            for name, model in models.items():
                prediction = model.predict(manual_data_scaled)
                prediction_proba = model.predict_proba(manual_data_scaled)[0]
                st.subheader(name)
                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {prediction_proba[1]:.3f}")
                st.markdown("---")
