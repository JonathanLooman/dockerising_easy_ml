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
from sklearn.cluster import KMeans
import yaml
import matplotlib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression_model(y_true, y_pred, num_features):
    """
    Evaluate a regression model and present the metrics in a visually appealing way.

    Parameters:
    - y_true: Actual target values.
    - y_pred: Predicted target values.
    - num_features: Number of features used in the model.

    Returns:
    - metrics_df: A pandas DataFrame containing the evaluation metrics.
    """

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - num_features - 1))

    # Create a DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R2)', 'Adjusted R-squared'],
        'Value': [mae, mse, rmse, r2, adj_r2]
    })

    # Style the DataFrame for better visualization
    metrics_df.style.bar(subset=['Value'], align='mid', color=['#d65f5f', '#5fba7d'])

    return metrics_df
def save_catboost_model(model, file_path):
    """
    Save a CatBoost model to a specified file path.

    Parameters:
    - model: The CatBoost model to be saved.
    - file_path: The file path where the model will be saved.

    Returns:
    - None
    """
    try:
        model.save_model(file_path)
        st.write(f"CatBoost model saved")
    except Exception as e:
        print(f"Error saving the model: {e}")



def multiclass_confision_matrix(y_test, y_pred):
    # Create a confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    # Create a classification report
    report = classification_report(y_test, y_pred)

    # Create subplots for the confusion matrix and classification report
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the confusion matrix as a heatmap in the first subplot
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    # Plot the classification report as text in the second subplot
    axes[1].text(0.1, 0.1, report, fontsize=12)
    axes[1].axis('off')
    axes[1].set_title('Classification Report')

   
    st.pyplot(fig)

def data_cleaning(df, method):
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

def multi_class_metrics(y_true, y_pred):
    """
    Calculate and display precision, recall, and F1-score for multi-class classification.

    Parameters:
    - y_true: The true class labels.
    - y_pred: The predicted class labels.

    Returns:
    - None (prints the metrics).
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-score: {f1:.4f}")
def train_model(df):



    
    output_variable = st.selectbox(
        'Select Dependent Variable',
        df.columns)
    st.write('Select which features to use in prediction')
    columns_used = []
    potential_independent_variables = [x for x in df.columns if x!=output_variable]
    for column_name in df.columns:
        if column_name!=output_variable:
            columns_used.append(st.checkbox(f'{column_name}'))
        else:
            columns_used.append(False)
    
    if df.isnull().sum().sum() != 0:
        st.write("There are null values in dataset")
        st.write(np.round(((df.isna().sum()/df.shape[0])*100), 0).reset_index(name='Percentage of missing values'))
        cleaning_method = st.selectbox(
        'Select method of data cleaning',
        ("drop_all_na", "fill_most_common"))
        

    model_type = st.selectbox(
    'Select type of model',
        ('Regression', 'Binary Classification', 'Multiclass Classification'))
    
    data = {
        'model_type': model_type,
        'output_variable': output_variable,
        'cleaning_method': cleaning_method,
        'columns_used':columns_used
    }
    with open('modelconfig.yaml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    train_test_split_per  = st.slider(
        'Select Percentage of data to be used for training',
        0.0, 100.0, 80.0)
    if st.button("Train Model"):
    
        with st.spinner("Training Model"):
            #df = df.dropna(subset='any')
            
            df_train = df.loc[np.random.choice(df.index, int(np.round(len(df)*(train_test_split_per/100), 0))),:]
            df_test = df[~df.index.isin(df_train.index.values)]

            df_samp_int = np.random.choice(df_train.loc[:, columns_used].index, 1000)
            x_samp = df_train.loc[df_samp_int, columns_used]
            
            x_samp = x_samp[~x_samp.isna().T.any()]
            x_samp.to_csv('temp_sample.csv')

            if df_train.isna().sum().max()/len(df_train)<0.03:
                df_train = df_train.dropna(how='any')
            else:
                df_train = df_train.fillna(df.mode().iloc[0])


            X_train = df_train.loc[:, columns_used]  # Replace 'target_column_name' with the actual target column name
            y_train = df_train[output_variable]

            X_test = df_test.loc[:, columns_used]  # Replace 'target_column_name' with the actual target column name
            y_test = df_test[output_variable]

            y_test = y_test[~X_test.isna().T.any()]
            X_test = X_test[~X_test.isna().T.any()]


            

            # Split the data into training and testing sets (adjust the test_size as needed)
            #test_size_temp = 1-(train_test_split_per/100)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_temp, random_state=42)
            # Automatically detect categorical columns using CatBoost's "auto" option
            
            categorical_features =  X_train.select_dtypes(include='object').columns.values
            if model_type =='Binary Classification':

                # Initialize the CatBoost classifier
                clf = CatBoostClassifier(iterations=1000,  # You can adjust the number of iterations
                                        depth=6,  # You can adjust the tree depth
                                        learning_rate=0.1,  # You can adjust the learning rate
                                        loss_function='Logloss',  # You can choose different loss functions
                                        cat_features=categorical_features)  # Specify the indices of categorical features

                # Train the classifier on the training data
                clf.fit(X_train, y_train)

                # Make predictions on the test data
                y_pred = clf.predict(X_test)

                # Evaluate the classifier's performance
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)
                save_catboost_model(clf, 'catboost_model.cbm')
                multiclass_confision_matrix(y_test, y_pred)


                return clf

            elif model_type =='Regression':
                # Create a CatBoost Pool for efficient data handling
                train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
                test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

                # Initialize the CatBoost regressor
                regressor = CatBoostRegressor(iterations=1000,  # You can adjust the number of iterations
                                            depth=6,  # You can adjust the tree depth
                                            learning_rate=0.1,  # You can adjust the learning rate
                                            loss_function='RMSE')  # Use RMSE for regression

                # Train the regressor
                regressor.fit(train_pool, verbose=100)  # Verbose setting for progress updates

                # Make predictions on the test data
                y_pred = regressor.predict(test_pool)

                # Evaluate the regression model's performance (e.g., RMSE)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print("Root Mean Squared Error (RMSE):", rmse)
                save_catboost_model(regressor, 'catboost_model.cbm')
                num_features = len(X_train.columns)  # Replace with the actual number of features used in your model

                metrics_df = evaluate_regression_model(y_test, y_pred, num_features)
                st.write(metrics_df)
                return regressor

            elif model_type == 'Multiclass Classification':

                # Initialize the CatBoost classifier for multi-output classification
                clf = CatBoostClassifier(iterations=1000,  # You can adjust the number of iterations
                                depth=6,  # You can adjust the tree depth
                                learning_rate=0.1,  # You can adjust the learning rate
                                loss_function='MultiClass',  # Use 'MultiClass' for multi-output classification
                                cat_features=categorical_features) 
                # Train the classifier on the training data
                clf.fit(X_train, y_train)

                # Make predictions on the test data
                y_pred = clf.predict(X_test)

                multi_class_metrics(y_test, y_pred)

                save_catboost_model(clf, 'catboost_model.cbm')

                multiclass_confision_matrix(y_test, y_pred)

                return clf

                
st.title('Easy ML Model Trainer')

#if st.button("Upload custom dataset"):
    #uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])
    #if uploaded_file is not None:
        #df = pd.read_csv(uploaded_file)
        
        #st.write(df.head(3))
        #prompt = st.text_area("Enter your prompt:")
#on = st.toggle('Activate feature')
#columns_used = []
#for column_names in df.columns:
    #columns_used.append(st.button(f'{column_names}'))

#df = pd.read_csv('sample_data\Airline Dataset.csv')
#st.write("Sample Data set on Airlines")
#st.write(df.head(3))

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(df.head(3))

if 'df' in globals():
    variable_in_question = st.selectbox(
        'Select a Variable to analyse',
        df.columns)
    

    if df[variable_in_question].dtype=='float':
        
        st.write(df[variable_in_question].describe())


        # Create subplots with 1 row and 2 columns
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    
        ax = sns.histplot(data=df, x=variable_in_question, kde=True, color='#F69521')
        
        for i in ax.containers:
    
            ax.bar_label(i, label_type='edge')
        # Set title and labels
        axes.set_title(f'Distribution of {variable_in_question}', fontsize=18)
        axes.set_xlabel(variable_in_question, fontsize=12)
        axes.set_ylabel('Count', fontsize=12)
    
        st.pyplot(fig)

    #elif df[variable_in_question].dtype!='float':
    else:
        #st.write(df[variable_in_question].describe())
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))


        ax = sns.countplot(data=df, x=variable_in_question, color='#F69521')
        for i in ax.containers:
            ax.bar_label(i, label_type='edge')


        # Set title and labels
        axes.set_title(f'Distribution of {variable_in_question}')
        axes.set_xlabel(variable_in_question)
        axes.set_ylabel('Count')

        st.pyplot(fig)

if 'df' in globals():
    clf = train_model(df)
if 'clf' in globals():    
    st.write('Please proceed to either Shap Variable Analyser to understand how features affect the predictions, \n or to Predictions to use the model to make individual and batch predictions')




