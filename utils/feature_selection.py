from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def select_k_best(X, y, k=10):
    """
    SelectKBest (f_regression): возвращает X_new, селектор, имена выбранных признаков.
    """
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    feature_names = list(X.columns[mask])
    return X_new, selector, feature_names


def rfe_selection(model, X, y, n_features=10):
    """
    RFE: возвращает X_new, селектор, имена выбранных признаков.
    """
    n_features = min(n_features, X.shape[1])
    selector = RFE(model, n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    feature_names = list(X.columns[mask])
    return X_new, selector, feature_names


def apply_pca(X, n_components=5):
    """
    PCA поверх стандартизованных признаков (устойчиво к масштабам).
    Возвращает X_pca и pipeline (StandardScaler -> PCA) с .transform().
    """
    n_components = min(n_components, X.shape[1])
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=n_components, random_state=42, svd_solver="full")
    )
    X_pca = pipe.fit_transform(X)
    return X_pca, pipe
