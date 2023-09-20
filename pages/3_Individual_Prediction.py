import streamlit as st
import time
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import yaml

st.title('Individual Prediction Analyser')

def load_catboost_model(model_path, model_type='classifier'):
    """
    Load a CatBoost model from a specified file.

    Parameters:
    - model_path: The file path where the CatBoost model is saved.
    - model_type: A string specifying the type of the model ('classifier' or 'regressor').

    Returns:
    - model: The loaded CatBoost model.
    """
    if model_type == 'classifier':
        model = CatBoostClassifier()
    elif model_type == 'regressor':
        model = CatBoostRegressor()
    else:
        raise ValueError("Invalid model_type. Use 'classifier' or 'regressor'.")

    model.load_model(model_path)
    return model

with open('modelconfig.yaml', 'r') as yaml_file:
    loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)


try:
    if loaded_data['model_type']=='Regression':
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='regressor')
    else:
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='classifier')
except:
    st.write("Please first train a model in Model Trainer")


st.write("Explore how our machine learning model makes predictions and gain insights into its decision-making process. To get started please select values below:")

df_samp = pd.read_csv("temp_sample.csv")
df_samp = df_samp.drop('Unnamed: 0',axis=1)

column_dict = {}
for column_name in df_samp.columns:
    data_type = df_samp[column_name].dtype
    if data_type == 'float64':
        float_var  = st.slider(
        column_name,
        df_samp[column_name].min(), df_samp[column_name].max(), df_samp[column_name].median())
        column_dict[column_name] = float_var
    elif data_type == 'O':
        object_var = st.selectbox(column_name,
            df_samp[column_name].unique())
        column_dict[column_name] = object_var
    elif data_type == 'int64':
        float_var  = st.slider(
        column_name,
        df_samp[column_name].min(), df_samp[column_name].max(), df_samp[column_name].mode()[0])
        column_dict[column_name] = float_var
    else:
        pass
# Convert the dictionary to a one-element DataFrame
df_pred = pd.DataFrame.from_dict(column_dict, orient='index').T
df_pred2 = df_pred[df_samp.columns]
prediction = clf.predict(df_pred2)
st.write(f"The prediction is: **{prediction[0]}**")
if loaded_data['model_type']=='Regression':
    pass
else:
    
    st.write(f"The probability of Positive is: **{np.round(clf.predict_proba(df_pred2)[0][1]*100,2)}%**")


# Create the explainer
explainer = shap.TreeExplainer(clf)
# Compute SHAP values
shap_values = explainer.shap_values(df_pred2)

fig_pred = shap.decision_plot(prediction, shap_values, df_pred2)
st.pyplot(fig_pred)
# model_type = st.selectbox(
#     'Select type of model',
#         ('Regression', 'Binary Classification', 'Multiclass Classification'))
# for column_name in df_samp.columns:
#     columns_used.append(st.checkbox(f'{column_name}'))
