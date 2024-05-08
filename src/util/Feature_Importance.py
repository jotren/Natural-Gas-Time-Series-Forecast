import torch
import numpy as np
import matplotlib.pyplot as plt

def rank_feature_importance(model, feature_names):
    """
    Rank feature importance for a given model.
    
    Parameters:
        - model: Trained machine learning model
        - feature_names: List of feature names
        
    Returns:
        - feature_importance: Dictionary mapping feature names to their importance scores
        - ranked_features: List of feature names sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (e.g., Random Forest, XGBoost)
        feature_importance = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # For linear models (e.g., Linear Regression)
        feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
    else:
        raise ValueError("Model does not support feature importances.")
    
    # Normalize feature importances
    total_importance = sum(feature_importance.values())
    feature_importance = {feat: importance / total_importance for feat, importance in feature_importance.items()}
    
    # Rank features by importance
    ranked_features = sorted(feature_importance, key=feature_importance.get, reverse=True)
    
    return feature_importance, ranked_features

def plot_feature_importance(feature_importance):
    """
    Plot feature importance.
    
    Parameters:
        - feature_importance: Dictionary mapping feature names to their importance scores
    """
    features, importance = zip(*sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importance, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance')
    plt.show()

def get_neural_network_feature_importance(model, feature_names):
    """
    Get feature importance for a neural network model (LSTM or GRU).
    
    Parameters:
        - model: Trained neural network model
        - feature_names: List of feature names
        
    Returns:
        - feature_importance: Dictionary mapping feature names to their importance scores
        - ranked_features: List of feature names sorted by importance
    """
    if isinstance(model, torch.nn.LSTM) or isinstance(model, torch.nn.GRU):
        input_weights = torch.abs(model.weight_ih_l0.detach().cpu().numpy())
        output_weights = torch.abs(model.weight_hh_l0.detach().cpu().numpy())
        
        # Normalize weights
        input_weights /= np.sum(input_weights, axis=0)
        output_weights /= np.sum(output_weights, axis=1, keepdims=True)
        
        # Compute overall importance scores by summing the weights across all time steps
        input_importance = np.sum(input_weights, axis=0)
        output_importance = np.sum(output_weights, axis=1)
        
        # Combine input and output importance into a single dictionary
        feature_importance = {}
        for i, feat_name in enumerate(feature_names):
            feature_importance[feat_name] = input_importance[i]
        
        # Rank features by importance
        ranked_features = sorted(feature_importance, key=feature_importance.get, reverse=True)
        
        return feature_importance, ranked_features
    else:
        raise ValueError("Model must be an instance of torch.nn.LSTM or torch.nn.GRU.")
