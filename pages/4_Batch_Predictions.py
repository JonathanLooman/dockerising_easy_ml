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

def data_cleaning(df, method='fill_most_common'):
    """
    Clean a DataFrame based on the specified method.

    Parameters:
    - df: The input DataFrame.
    - method: A string specifying the cleaning method.
        - "drop_all_na": Remove rows with null values.
        - "fill_most_common": Replace null values with the most common value in each column.

    Returns:
    - cleaned_df: The cleaned DataFrame.
    - null_values_removed: True if all null values have been removed, False otherwise.
    """
    # Find all object (string) columns
    object_columns = df.select_dtypes(include='object')
    for colname in object_columns:
        df[[x.lower()=='null' for x in df[colname].astype(str)]] = np.nan
    if method == "drop_all_na":
        cleaned_df = df.dropna()

    elif method == "fill_most_common":
        cleaned_df = df.fillna(df.mode().iloc[0])

    else:
        raise ValueError("Invalid method. Use 'drop_all_na' or 'fill_most_common'.")

    null_values_removed = cleaned_df.isnull().sum().sum() == 0
    st.write(f"All null values removed {null_values_removed}")
    return cleaned_df


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


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

with open('modelconfig.yaml', 'r') as yaml_file:
    loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)


try:
    if loaded_data['model_type']=='Regression':
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='regressor')
    else:
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='classifier')
except:
    st.write("Please first train a model in Model Trainer")

st.title('Batch Prediction Tool')
st.write("Please upload a dataset for the model to make predictions on. Please ensure the columns in the dataset are exactly the same as the data used to train the mode.")
uploaded_file = st.file_uploader("Please Upload a file to make predictions on", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = data_cleaning(df=df, method='fill_most_common')
    #st.write(df.head(5))

    
df_samp = pd.read_csv("temp_sample.csv")
df_samp = df_samp.drop('Unnamed: 0',axis=1)

if 'df' in globals():
    df_pred2 = df[df_samp.columns].copy()


    if loaded_data['model_type']=='Regression':
        prediction = clf.predict(df_pred2)
        df.loc[:,'prediction'] = np.reshape(prediction, (-1,1))
        
    elif loaded_data['model_type']=='Binary Classification':

        float_var  = st.slider(
            'Probability Threshold',
            0.00, 1.00, 0.50)
        clf.set_probability_threshold(float_var)
        prediction = clf.predict(df_pred2)

        df.loc[:,'prediction'] = np.reshape(prediction, (-1,1))
        df['Positive_probability'] = [x[1] for x in clf.predict_proba(df_pred2[df_samp.columns])]
        st.write(df)
        


    elif loaded_data['model_type']=='Multiclass Classification':
        
        
        prediction = clf.predict(df_pred2)

        df.loc[:,'prediction'] = np.reshape(prediction, (-1,1))
        
        #df_prob = clf.predict_proba(df_pred2[df_samp.columns])
        # Merge the DataFrames on their index
        
        # Use predict_proba to get class probabilities
        class_probabilities = clf.predict_proba(df_pred2)

        # Get the order of class labels
        class_labels = clf.classes_

        # Create a DataFrame to display results and probabilities
        result_df = pd.DataFrame(class_probabilities, columns=class_labels)
        
        df = df.merge(result_df, left_index=True, right_index=True)
        
       
        st.write(df)
    
    else:
        pass

    if st.button("Save File"):
        csv = convert_df(df)
        df.to_csv('temp_results_file.csv')


if 'csv' in globals():
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='output_file.csv',
    mime='text/csv',
)