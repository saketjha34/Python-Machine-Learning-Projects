import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def objective(trial, X_train ,train_targets ,X_val ,val_targets):

    param = {
        "objective": "multi:softmax",
        "metric": "multi_logloss",
        "boosting_type": "gbtree",
        'num_class':7,
        'eta': trial.suggest_float('learning_rate', 0.01, 0.05),
        "n_estimators": trial.suggest_int("n_estimators", 400,600, log=True),
        'max_depth': trial.suggest_int("max_depth", 6,20, log=True),
        'max_leaves' : trial.suggest_int("max_leaves", 2,20, log=True),
        'gamma': trial.suggest_float("gamma", 0.1,1 ,log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight':trial.suggest_int("min_child_weight", 10,40, log=True),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-1, 10.0),
        'min_split_gain': trial.suggest_float("min_split_gain", 0.4, 0.8, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.9, log=True),
    }
    model = XGBClassifier(n_jobs=-1,**param).fit(X_train,train_targets)
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(val_preds,val_targets)
    return accuracy