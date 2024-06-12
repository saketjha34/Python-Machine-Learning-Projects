import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

def objective(trial,X_train ,train_targets ,X_val ,val_targets):

    param = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20,60, log=True),
        "bagging_freq": trial.suggest_int("bagging_freq", 1,10, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300,800, log=True),
        'max_depth': trial.suggest_int("max_depth", 10,40, log=True),
        'learning_rate':trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5,20, log=True),
        'min_child_weight':trial.suggest_float("min_child_weight", 1e-5,1e-2, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-1,0.8, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-1, 0.8, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8),
        'min_split_gain': trial.suggest_float("min_split_gain", 0.4, 0.9, log=True),
        'colsample_bytree ': trial.suggest_float("colsample_bytree ", 0.3, 0.9, log=True),
    }

    model = LGBMClassifier(n_jobs=-1,**param).fit(X_train,train_targets)
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(val_preds,val_targets)
    return accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    trial.params

    def plots():
        fig = optuna.visualization.plot_slice(study, params= trial.params.keys())
        fig.show()  
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(width=900,height=600)
        fig.show()