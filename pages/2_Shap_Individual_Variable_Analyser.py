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

st.title('Shap Varibale Analyzer')

from catboost import CatBoostClassifier, CatBoostRegressor

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

df_samp = pd.read_csv("temp_sample.csv")
df_samp = df_samp.drop('Unnamed: 0',axis=1)

st.set_option('deprecation.showPyplotGlobalUse', False)



variable1 = st.selectbox(
        'Select Variable to Analyse',
        df_samp.columns)

# Create the explainer
explainer = shap.TreeExplainer(clf)

# Compute SHAP values
shap_values = explainer.shap_values( df_samp)

# If shap_values is a list (for classification), select the values for the desired class

fig = shap.dependence_plot(variable1, shap_values, df_samp )
st.pyplot(fig)