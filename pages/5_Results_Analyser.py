import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
import yaml
import matplotlib


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

def data_cluster(df_new, cluster_column):
    """
    Segment the DataFrame into an optimal number of 
    groups based on 'Positive_probability'.
   

    Parameters:
    df_new (DataFrame): The input DataFrame.

    Returns:
    df_new (DataFrame): The input DataFrame with an additional 
    column named "Cluster" containing the number of the 
    cluster assigned
    """
    # Extract the 'Positive_probability' column as a feature for clustering
    features = df_new[[cluster_column]].values

    # Determine the optimal number of clusters using the Elbow method
    distortions = []
    K_range = range(1, 11)  # Try a range of cluster numbers
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)
    
    # Find the "elbow point" to determine the optimal number of clusters
    elbow_point = np.argmin(np.diff(distortions)) + 1
    optimal_clusters = K_range[elbow_point]

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    df_new['Cluster'] = kmeans.fit_predict(features)
    return df_new


# Create a function to filter out probabilities that were not predicted
def filter_probabilities(results):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(results)
    
    # Initialize an empty DataFrame to store filtered results
    filtered_df = pd.DataFrame(columns=df.columns)
    
    # Iterate through rows
    for index, row in df.iterrows():
        # Get the predicted class for the current row
        prediction = row['prediction']
        
        # Filter out probabilities that were not predicted for
        filtered_row = {col: prob for col, prob in row.items() if col == prediction or col == 'prediction'}
        
        # Append the filtered row to the result DataFrame
        filtered_df = pd.concat([filtered_df, pd.DataFrame(filtered_row, index=[index])], ignore_index=True)
    
    return filtered_df


def plot_binary_results(df_new):
    # Set a custom color palette
    custom_palette = ['#1f77b4', '#ff7f0e']

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot a histogram of 'Positive_probability' in the first subplot
    sns.histplot(data=df_new, x='Positive_probability', bins=20, ax=axes[0], color=custom_palette[0], kde=True)
    axes[0].set_title('Histogram of Positive Probability')
    axes[0].set_xlabel('Positive Probability')
    axes[0].set_ylabel('Frequency')

    # Plot a bar chart of predicted values [0.0, 1.0] in the second subplot
    sns.countplot(data=df_new, x='prediction', ax=axes[1], order=[0.0, 1.0], palette=custom_palette)
    axes[1].set_title('Bar Chart of Predicted Values')
    axes[1].set_xlabel('Predicted Value')
    axes[1].set_ylabel('Count')

    # Customize grid appearance
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for better visualization
    plt.tight_layout()

    st.pyplot(fig)

def plot_multiclass_results(df_new):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set a custom color palette
    custom_palette = sns.color_palette('Set2', n_colors=len(df_new['prediction'].unique()))

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Plot a histogram of class probabilities in the first subplot
    ax = sns.countplot(data=df_new, x='prediction',  color=custom_palette[0])
    for i in ax.containers:
        ax.bar_label(i, label_type='edge')
    
    axes[0].set_title('Bar Plot of Number of Predictions')
    axes[0].set_xlabel('Class predicted')
    axes[0].set_ylabel('Number of Predictions')



    # Find the index of the "predictions" column
    predictions_index = df.columns.get_loc("prediction")
    # Slice the DataFrame to get columns after "predictions"
    columns_after_predictions = df_new.iloc[:, predictions_index :]

    columns_after_predictions['prediction'] = [str(x) for x in columns_after_predictions['prediction']]

    # Call the function to get filtered results
    filtered_results = filter_probabilities(columns_after_predictions)

    sns.boxplot(data=filtered_results)
    axes[1].set_title('Box plot showing probabilities of each prediction')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Probability')

    # Customize grid appearance
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for better visualization
    plt.tight_layout()

    st.pyplot(fig)


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
st.title('Result Analyser Tool')
st.write("Below is distribution of predictions made by the model.")


with open('modelconfig.yaml', 'r') as yaml_file:
    loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)


try:
    if loaded_data['model_type']=='Regression':
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='regressor')
    else:
        clf = load_catboost_model(model_path='catboost_model.cbm', model_type='classifier')
except:
    st.write("Please first train a model in Model Trainer")

df = pd.read_csv('temp_results_file.csv')
df = df.drop('Unnamed: 0', axis=1)

if loaded_data['model_type']== 'Binary Classification':
    plot_binary_results(df)
elif loaded_data['model_type']== 'Multiclass Classification':
    plot_multiclass_results(df)


    #if st.button("Cluster based on Positive Probability"):
#     cluster_column = st.selectbox(
#     'Select column to cluster dataset',
#         df.columns)
#     df_new = data_cluster(df, cluster_column)

#     #if st.button("Pofile based on Cluster"):
#     show_profile = True
        
# if 'show_profile' in globals(): 
#     st.write(df_new.groupby('Cluster').apply(lambda x: x.mode().iloc[0]))
