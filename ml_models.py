# ml_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, log_loss, accuracy_score,
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class MFIModelTrainer:
    """Train and evaluate multiple ML models for MFI loan risk assessment"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models with hyperparameter grids"""
        
        self.models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.5, 1.0, 1.5],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            },
            
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            }
        }
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """Train all models with hyperparameter tuning"""
        
        print("=== Training MFI Loan Risk Models ===\n")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model_config in self.models.items():
            print(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Store the best model
            self.trained_models[name] = grid_search.best_estimator_
            
            # Store results
            self.model_results[name] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            }
            
            print(f"  Best CV AUC: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_score'][grid_search.best_index_] * 2:.4f})")
            print(f"  Best params: {grid_search.best_params_}\n")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        
        print("=== Model Evaluation Results ===\n")
        
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            
            # Store results
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'log_loss': logloss,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Log Loss: {logloss:.4f}\n")
        
        # Find best model based on ROC-AUC
        best_model_name = max(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['roc_auc'])
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        print(f"Best Model: {best_model_name} (ROC-AUC: {evaluation_results[best_model_name]['roc_auc']:.4f})")
        
        return evaluation_results
    
    def plot_model_comparison(self, evaluation_results):
        """Plot comparison of all models"""
        
        # Prepare data for plotting
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Individual metric plots
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Overall comparison radar chart style
        metrics_data = []
        for model in models:
            model_metrics = [evaluation_results[model][metric] for metric in metrics]
            metrics_data.append(model_metrics)
        
        # Heatmap of all metrics
        axes[5].remove()
        ax_heatmap = fig.add_subplot(2, 3, 6)
        
        heatmap_data = pd.DataFrame(metrics_data, 
                                  index=models, 
                                  columns=[m.replace('_', ' ').title() for m in metrics])
        
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', 
                   cbar_kws={'label': 'Score'}, ax=ax_heatmap)
        ax_heatmap.set_title('Model Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=(12, 8))
        
        for name, model in self.trained_models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
