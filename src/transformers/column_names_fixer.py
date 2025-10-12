def fix_feature_names(X):
    if hasattr(X, 'columns'):
        X.columns = [c.split('__')[-1] for c in X.columns]
    return X
