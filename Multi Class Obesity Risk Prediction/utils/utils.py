import seaborn as sns
import matplotlib.pyplot  as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score , f1_score , roc_auc_score
from joblib import Parallel, delayed
import optuna
from sklearn.base import ClassifierMixin
from typing import Type


models:dict[str, dict[str,float]] = {
    LogisticRegression : {
        'n_jobs' : -1
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
    DecisionTreeClassifier : {
        'n_estimators': 500,
        'num_leaves': 32 , 
        'learning_rate': 0.01,
        'subsample': 0.7,
    },
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

    train_f1 = f1_score(train_targets, train_preds)
    val_f1 = f1_score(val_targets, val_preds)

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

def plot_model_importance(model: Type[ClassifierMixin], 
                          X_train: pd.DataFrame,
                          train_targets: pd.Series,
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
    classifier = model(**params)
    classifier.fit(X_train, train_targets)
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 30))
    plt.title('Feature Importance of model')
    sns.barplot(data=importance_df, x='importance', y='feature', color='blue')
    plt.show()
    
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
