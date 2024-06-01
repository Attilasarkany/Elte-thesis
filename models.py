import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from sklearn.metrics import balanced_accuracy_score, log_loss, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV




def Logistic(X_train, X_test, y_train, y_test,feature_names,C):
    model = LogisticRegression(class_weight='balanced',penalty='l2', C=C)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions and get probabilities on the test data
    preds = model.predict(X_test)
    pred_probas = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Compute balanced accuracy
    tr_preds = model.predict(X_train)
    train_bal_acc = balanced_accuracy_score(y_train, tr_preds)
    test_bal_acc = balanced_accuracy_score(y_test, preds)
    log_loss_val = log_loss(y_test, pred_probas)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probas, pos_label=1)
    auc_score = auc(fpr, tpr)

    # Calculate percentage of negative predictions
    percentage_neg_preds = np.sum(preds == 0) / len(preds)

    # Generate confusion matrix and plot
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    feature_importance = np.abs(model.coef_[0])
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)



    # Summary dictionary
    summary = {
        'AUC': auc_score,
        'Train Balanced Accuracy': train_bal_acc,
        'Test Balanced Accuracy': test_bal_acc,
        'Percentage of Negative Predictions': percentage_neg_preds,
        'Log Loss': log_loss_val,
        'Confusion Matrix': cm,
        'FPR (False Positive Rate)': fpr,
        'TPR (True Positive Rate)': tpr,
        'Thresholds': thresholds,
        'Model': model,
        'Feature Importance': importance_df
    }



    return preds, tr_preds,summary


def rf(X_train, X_test, y_train, y_test,feature_names,max_depth):
    model = RandomForestClassifier(class_weight='balanced', max_depth=max_depth)
    model.fit(X_train, y_train)

    # Make predictions and compute probabilities on the test data
    preds = model.predict(X_test)
    pred_probas = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Calculate metrics
    tr_preds = model.predict(X_train)
    train_bal_acc = balanced_accuracy_score(y_train, tr_preds)
    test_bal_acc = balanced_accuracy_score(y_test, preds)
    log_loss_val = log_loss(y_test, pred_probas)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probas, pos_label=1)
    auc_score = auc(fpr, tpr)

    # Calculate percentage of negative predictions
    percentage_neg_preds = np.sum(preds == 0) / len(preds)

    # Generate confusion matrix and plot
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

    # Summary dictionary
    summary = {
        'AUC': auc_score,
        'Train Balanced Accuracy': train_bal_acc,
        'Test Balanced Accuracy': test_bal_acc,
        'Percentage of Negative Predictions': percentage_neg_preds,
        'Log Loss': log_loss_val,
        'Confusion Matrix': cm,
        'Feature Importance': importance_df,
        'FPR (False Positive Rate)': fpr,
        'TPR (True Positive Rate)': tpr,
        'Thresholds': thresholds,
        'Model': model
    }

 
    return preds, tr_preds, summary

def lightGBM(X_train, X_test, y_train, y_test,feature_names, max_depth=7):
    model = LGBMClassifier(
        max_depth=max_depth,
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
        metric='binary_logloss'  # Equivalent to 'logloss' in XGBoost
    )

    model.fit(X_train, y_train)

    # Make predictions and compute probabilities on the test data
    preds = model.predict(X_test)
    pred_probas = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Calculate metrics
    tr_preds = model.predict(X_train)
    train_bal_acc = balanced_accuracy_score(y_train, tr_preds)
    test_bal_acc = balanced_accuracy_score(y_test, preds)
    log_loss_val = log_loss(y_test, pred_probas)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probas, pos_label=1)
    auc_score = auc(fpr, tpr)

    # Calculate percentage of negative predictions
    percentage_neg_preds = np.sum(preds == 0) / len(preds)

    # Generate confusion matrix and plot
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

    # Summary dictionary
    summary = {
        'AUC': auc_score,
        'Train Balanced Accuracy': train_bal_acc,
        'Test Balanced Accuracy': test_bal_acc,
        'Percentage of Negative Predictions': percentage_neg_preds,
        'Log Loss': log_loss_val,
        'Confusion Matrix': cm,
        'FPR (False Positive Rate)': fpr,
        'TPR (True Positive Rate)': tpr,
        'Thresholds': thresholds,
        'Feature Importance': importance_df,

        'Model': model
    }

    return preds, tr_preds, summary


def time_series_grid_search(model_choice, X_train, X_test, y_train, y_test, feature_names, n_splits=3):
    from sklearn.metrics import fbeta_score, make_scorer, confusion_matrix, roc_curve, auc

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
    
    model, param_grid = None, {}

    if model_choice == 'logistic':
        model = LogisticRegression(class_weight='balanced', penalty='l2')
        param_grid = {'C': [0.001, 0.0001]}
    elif model_choice == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced')
        param_grid = {'n_estimators': [100, 150], 'max_depth': [7, 8]}
    elif model_choice == 'lightgbm':
        model = LGBMClassifier(scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])))
        param_grid = {'num_leaves': [31, 40], 'max_depth': [7, 8]}
    else:
        raise ValueError("Model choice not supported")

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring=scorer)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    preds = best_model.predict(X_test)
    pred_probas = best_model.predict_proba(X_test)[:, 1]
    test_balanced_acc_score = balanced_accuracy_score(y_test, preds)  # Calculate balanced accuracy for test set
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probas, pos_label=1)
    auc_score = auc(fpr, tpr)

    # Display results
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Best Model')
    plt.show()

    # Feature Importance
    feature_importance = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else np.abs(best_model.coef_[0])
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

    # Return results including test set balanced accuracy
    return {
        'Best Model': best_model,
        'Best Score': grid_search.best_score_,
        'Best Params': grid_search.best_params_,
        'Confusion Matrix': cm,
        'AUC Score': auc_score,
        'Balanced Accuracy Score (Test)': test_balanced_acc_score,
        'Feature Importance': importance_df
    }

