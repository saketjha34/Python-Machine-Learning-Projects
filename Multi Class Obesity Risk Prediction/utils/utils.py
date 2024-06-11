import seaborn as sns
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import optuna
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score , f1_score , confusion_matrix ,classification_report



models:dict[str, dict[str,float]] = {
    
    GaussianNB : {

    },

    LogisticRegression : {
        'n_jobs' : -1
    },

    SVC: {
        'C' : 10,
        'kernel': 'rbf',
        
    },
    
    SGDClassifier : {
        'eta0' : 0.001 ,
        'max_iter': 2000 ,
        'penalty': 'l1' ,
        'learning_rate': 'adaptive'
    },

    RandomForestClassifier : {
        'max_depth': 32 , 
        'max_features': 'sqrt' , 
        'max_leaf_nodes': 100 , 
        'min_samples_leaf': 4
    },

    GradientBoostingClassifier : {
        'learning_rate': 0.001,
        'n_estimators' : 400,

    },

    DecisionTreeClassifier:{

    },

    XGBClassifier : {
                'n_estimators': 400 ,
                'max_depth': 32 , 
                'min_samples_split': 10 ,
                'min_samples_leaf':4 , 
                'n_jobs':-1
    },

    LGBMClassifier : {
        'n_estimators': 400,
        'max_depth': 10,
        'learning_rate': 0.01,
        'subsample': 0.5,

    },

    MLPClassifier : {
        'activation': 'identity',
        'solver': 'adam',
        'batch_size': 32,
        'max_iter': 400,
        'random_state': 42,
        'beta_1': 0.5,

    },
    CatBoostClassifier :{
    'n_estimators': 900,
    'verbose': 0 ,
    'learning_rate': 0.005,
    'l2_leaf_reg':1e-3,
    }
}


def evaluate_model(model,
                   params: dict,
                   X_train: np.ndarray,
                   train_targets: np.ndarray,
                   X_val: np.ndarray, 
                   val_targets: np.ndarray) -> tuple[float, float, float, float]:
    """
    Trains a model and evaluates it on training and validation datasets.

    Parameters:
    - model: The regressor class to be used for training.
    - params (Dict): Dictionary of parameters to initialize the model.
    - X_train (np.ndarray): The training data features.
    - train_targets (np.ndarray): The training data targets.
    - X_val (np.ndarray): The validation data features.
    - val_targets (np.ndarray): The validation data targets.

    Returns:
    - Tuple[float, float, float, float]: A tuple containing training accuracy,
      validation accuracy, training F1 score, and validation F1 score.
    """
    regressor = model(**params).fit(X_train, train_targets)
    train_preds = regressor.predict(X_train)
    val_preds = regressor.predict(X_val)

    train_acc = accuracy_score(train_targets, train_preds)
    val_acc = accuracy_score(val_targets, val_preds)

    train_f1 = f1_score(train_targets, train_preds , average='macro')
    val_f1 = f1_score(val_targets, val_preds , average='macro')

    return (train_acc, val_acc, train_f1, val_f1)


def try_models(model_dict: dict[str, dict[str,float]], 
               X_train: np.ndarray, 
               train_targets: np.ndarray, 
               X_val: np.ndarray, 
               val_targets: np.ndarray) -> pd.DataFrame:
    """
    Evaluates multiple models on training and validation datasets.

    Parameters:
    - model_dict (Dict): A dictionary where keys are model classes and values are dictionaries of parameters.
    - X_train (np.ndarray): The training data features.
    - train_targets (np.ndarray): The training data targets.
    - X_val (np.ndarray): The validation data features.
    - val_targets (np.ndarray): The validation data targets.

    Returns:
    - pd.DataFrame: A dataframe containing model names, parameters, and evaluation metrics.
    """
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(model, params, X_train, train_targets, X_val, val_targets) for model, params in model_dict.items())

    metrics = ['train_acc', 'val_acc', 'train_f1', 'val_f1']
    results_dict = {metric: [result[i] for result in results] for i, metric in enumerate(metrics)}

    df = pd.DataFrame({
        'models': list(model_dict.keys()),
        'params': list(model_dict.values()),
        **results_dict
    })

    return df


def evalmodel(model,
              X_train:np.array , 
              train_targets:np.array ,
              X_val:np.array,
              val_targets:np.array,
              **params) -> dict[str , float]:
    """
    Trains a given model with training data and evaluates its performance on both training and validation data.

    Parameters:
    model (class): A machine learning model class (e.g., sklearn.ensemble.RandomForestClassifier).
    X_train (np.array): Feature matrix for training data.
    train_targets (np.array): Target values for training data.
    X_val (np.array): Feature matrix for validation data.
    val_targets (np.array): Target values for validation data.
    **params: Additional parameters to initialize the model.

    Returns:
    dict[str, float]: A dictionary containing the accuracy and F1 score for both training and validation data.
    """
    model = model(**params).fit(X_train,train_targets)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    return {
        'Train Accuracy:': accuracy_score(train_preds,train_targets ),
        'Val Accuracy:' :  accuracy_score(val_preds,val_targets ),
        'Train F1 Score:' :  f1_score(val_preds,val_targets, average='macro'),
        'Val F1 Score:' :  f1_score(val_preds,val_targets , average='macro'),
    }

def submit_prediction(model,
                      submission_df: pd.DataFrame, 
                      X_train: np.ndarray, 
                      train_targets: np.ndarray,
                      X_test: np.ndarray, 
                      index: int,
                      **params:dict[str, float] ) -> dict[str, float]:
    """
    Trains a model, makes predictions on test data, and saves the predictions to a CSV file.

    Parameters:
    - model: The model class to be used for training.
    - submission_df (pd.DataFrame): The submission dataframe to store the predictions.
    - X_train (np.ndarray): The training data features.
    - train_targets (np.ndarray): The training data targets.
    - X_test (np.ndarray): The test data features.
    - index (int): An index used to differentiate the submission file.
    - **params: Additional parameters to initialize the model.

    Returns:
    - Dict[str, float]: A dictionary containing the parameters used to initialize the model, 
      including the trained model instance.
    """
    model = model(**params).fit(X_train, train_targets)
    test_preds = model.predict(X_test)
    submission_df['price'] = test_preds
    submission_path = Path('submission')
    submission_df.to_csv(f'{submission_path}\\submission_df{index}.csv', index=False)
    params['model'] = model
    return params

def plot_model_importance(model: type[ClassifierMixin], 
                          X_train: pd.DataFrame, 
                          train_targets: pd.Series ,
                          get_importance_df : bool = False,
                          **params) -> pd.DataFrame:
    """
    Trains a model and plots the feature importances.

    Parameters:
    - model (Type[ClassifierMixin]): The classifier to be used for training.
    - X_train (pd.DataFrame): The training data features.
    - train_targets (pd.Series): The training data targets.
    - **params: Additional parameters to be passed to the model.

    Returns:
    - pd.DataFrame: A dataframe containing feature names and their importance scores.
    """
    # Train the model
    classifier = model(**params)
    classifier.fit(X_train, train_targets)
    
    # Create a DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")
    barplot = sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    
    # Customize the plot appearance
    barplot.set_title('Feature Importance of Model', fontsize=16, weight='bold')
    barplot.set_xlabel('Importance', fontsize=14)
    barplot.set_ylabel('Feature', fontsize=14)
    barplot.tick_params(axis='x', rotation=90, labelsize=12)
    barplot.tick_params(axis='y', labelsize=12)
    
    # Add value labels on the bars
    for index, value in enumerate(importance_df['importance']):
        barplot.text(value, index, f'{value:.2f}', color='black', ha="left", va="center", fontsize=12)
    
    plt.tight_layout()
    plt.show()
    if get_importance_df == True:
        return importance_df


def try_models(model_dict: dict[str, dict[str,float]],
               X_train: np.ndarray,
               train_targets: np.ndarray,
               X_val: np.ndarray,
               val_targets: np.ndarray) -> pd.DataFrame:
    """
    Trains and evaluates multiple models on training and validation datasets.

    Parameters:
    - model_dict (dict[str, dict[str,float]]): A dictionary where keys are model classes and values are dictionaries of parameters.
    - X_train (np.ndarray): The training data features.
    - train_targets (np.ndarray): The training data targets.
    - X_val (np.ndarray): The validation data features.
    - val_targets (np.ndarray): The validation data targets.

    Returns:
    - pd.DataFrame: A dataframe containing model names, parameters, and evaluation metrics.
    """
    train_acc = []
    val_acc = []
    train_f1 = []
    val_f1 = []


    for model, key in model_dict.items():
        regressor = model(**key).fit(X_train, train_targets)
        train_preds = regressor.predict(X_train)
        val_preds = regressor.predict(X_val)

        train_acc.append(accuracy_score(train_targets, train_preds))
        val_acc.append(accuracy_score(val_targets, val_preds))

        train_f1.append(f1_score(train_targets, train_preds, squared=False))
        val_f1.append(f1_score(val_targets, val_preds, squared=False))

        
    df = pd.DataFrame({
        'models': list(model_dict.keys()),
        'params': list(model_dict.values()),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,

    })
    return df

def plot_confusion_matrix(model: type[ClassifierMixin], 
                          X_train: np.array, 
                          train_targets: np.array,
                          X_val: np.array = None, 
                          val_targets: np.array = None, 
                          labels: list = None,
                          **params) ->None :
    """
    Trains a model and plots the confusion matrices for both training and validation datasets.

    Parameters:
    - model (Type[ClassifierMixin]): The classifier to be used for training.
    - X_train (np.array): The training data features.
    - train_targets (np.array): The training data targets.
    - X_val (np.array, optional): The validation data features.
    - val_targets (np.array, optional): The validation data targets.
    - labels (list, optional): The list of labels to be used in the confusion matrix heatmap.
    - **params: Additional parameters to be passed to the model.
    """
    classifier = model(**params).fit(X_train, train_targets)
    train_preds = classifier.predict(X_train)
    train_cm = confusion_matrix(train_targets, train_preds)
    
    if X_val is not None and val_targets is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    sns.heatmap(train_cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                xticklabels=labels if labels is not None else 'auto', 
                yticklabels=labels if labels is not None else 'auto')
    axes[0].set_title('Confusion Matrix: Training Dataset')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')

    if X_val is not None and val_targets is not None:
        val_preds = classifier.predict(X_val)
        val_cm = confusion_matrix(val_targets, val_preds)
        
        sns.heatmap(val_cm, annot=True, fmt='d', ax=axes[1], cmap='rocket',
                    xticklabels=labels if labels is not None else 'auto', 
                    yticklabels=labels if labels is not None else 'auto')
        axes[1].set_title('Confusion Matrix: Validation Dataset')
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')
    
    plt.tight_layout()
    plt.show()

def plot_classification_report(model: type[ClassifierMixin], 
                               X_train: np.array, 
                               train_targets: np.array, 
                               X_val: np.array = None, 
                               val_targets: np.array = None, 
                               labels: list = None, 
                               **params) -> None:
    """
    Trains a model and prints the classification reports for both training and validation datasets.

    Parameters:
    - model (Type[ClassifierMixin]): The classifier to be used for training.
    - X_train (np.array): The training data features.
    - train_targets (np.array): The training data targets.
    - X_val (np.array, optional): The validation data features.
    - val_targets (np.array, optional): The validation data targets.
    - labels (list, optional): The list of labels to be used in the classification report.
    - **params: Additional parameters to be passed to the model.
    """
    classifier = model(**params).fit(X_train, train_targets)
    train_preds = classifier.predict(X_train)
    train_report = classification_report(train_targets, train_preds, target_names=labels)
    
    print('-----------------------------------------------------------')
    print('------------------Training Dataset Report------------------')
    print(train_report)
    
    if X_val is not None and val_targets is not None:
        val_preds = classifier.predict(X_val)
        val_report = classification_report(val_targets, val_preds, target_names=labels)
        
        print('-----------------------------------------------------------')
        print('----------------Validation Dataset Report------------------')
        print(val_report)



filepath = '../saved models/LGBModel.joblib'
def save_model(model : type[ClassifierMixin], 
               model_params : dict[str,float], 
               filepath:str) -> None:
    """
    Save a model and its parameters to a specified file path using joblib.

    Parameters:
    model (type[ClassifierMixin]) : The model to be saved.
    model_params (dict[str,float]) : The parameters used to create the model.
    filepath (str): The file path where the model and its parameters will be saved.
    
    Returns:
    None
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model and parameters to a dictionary
    model_data = {
        'model': model,
        'params': model_params
    }
    
    # Save the dictionary to the specified file path
    joblib.dump(model_data, filepath)
    print(f"Model and parameters saved to {filepath}")

def load_model(filepath:str) -> tuple[type[ClassifierMixin],dict[str,float]]:
    """
    Load a model and its parameters from a specified file path using joblib.

    Parameters:
    filepath (str): The file path from where the model and its parameters will be loaded.
    
    Returns:
    model (type[ClassifierMixin]) : The loaded model.
    model_params (dict[str,float]): The parameters used to create the model.
    """
    # Load the model and parameters from the specified file path
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    model_params = model_data['params']
    
    print(f"Model and parameters loaded from {filepath}")
    return model, model_params