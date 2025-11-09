from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA

def select_k_best(X, y, k=10):
    """
    Отбор признаков SelectKBest
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

def rfe_selection(model, X, y, n_features=10):
    """
    Рекурсивный отбор признаков RFE
    """
    selector = RFE(model, n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

def apply_pca(X, n_components=5):
    """
    Применение PCA для уменьшения размерности
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
