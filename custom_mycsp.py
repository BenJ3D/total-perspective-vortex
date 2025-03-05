import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=0.0, norm_trace=False, log=True):
        self.n_components = n_components
        self.reg = reg
        self.norm_trace = norm_trace
        self.log = log
        self.filters_ = None
        self.whitening_mat_ = None
        self.rank_ = None

    def _compute_covariance(self, X):
        """
        Calcule la matrice de covariance d'un essai EEG.

        Paramètres
        ----------
        X : ndarray, shape (n_channels, n_samples)

        Retourne
        -------
        cov : ndarray, shape (n_channels, n_channels)
        """
        return np.dot(X, X.T) / X.shape[1]

    def fit(self, X, y):
        """
        Entraîne le modèle CSP sur des données EEG.

        Paramètres
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape (n_trials,)

        Retourne
        -------
        self : instance de MyCSP
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"MyCSP est conçu pour 2 classes, {len(classes)} classes fournies.")

        covariances = {}
        for cls in classes:
            X_cls = X[y == cls]
            cov_list = []
            for trial in X_cls:
                cov_trial = self._compute_covariance(trial)
                if self.norm_trace:
                    cov_trial = cov_trial / np.trace(cov_trial)
                cov_list.append(cov_trial)
            cov_avg = np.mean(cov_list, axis=0)
            n_channels = cov_avg.shape[0]
            # Régularisation
            if self.reg:
                if isinstance(self.reg, (int, float)):
                    cov_avg += self.reg * np.trace(cov_avg) / n_channels * np.eye(n_channels)
                elif isinstance(self.reg, str) and self.reg.lower() == 'ledoit_wolf':
                    # Concatène tous les essais pour estimer la covariance via Ledoit-Wolf
                    X_concat = np.concatenate([trial for trial in X_cls], axis=1)
                    X_concat = X_concat.T  # forme (total_samples, n_channels)
                    cov_avg, _ = ledoit_wolf(X_concat)
                else:
                    raise ValueError(f"Option de régularisation inconnue: {self.reg}")
            covariances[cls] = cov_avg

        # Construction de la covariance composite
        cov_composite = covariances[classes[0]] + covariances[classes[1]]
        eigvals, eigvecs = np.linalg.eigh(cov_composite)
        # Calcul d'un seuil de tolérance comme dans MNE
        tol = np.finfo(eigvals.dtype).eps * cov_composite.shape[0] * np.max(eigvals)
        idx = eigvals > tol
        if not np.any(idx):
            raise ValueError("La covariance composite a un rang nul.")
        eigvals_red = eigvals[idx]
        eigvecs_red = eigvecs[:, idx]
        r = len(eigvals_red)
        if r < cov_composite.shape[0]:
            print(f"[INFO] Réduction du rang des données de {cov_composite.shape[0]} à {r}.")
        self.rank_ = r

        # Whitening dans l'espace réduit
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_red))
        whitening_mat = eigvecs_red @ D_inv_sqrt
        self.whitening_mat_ = whitening_mat

        # Projet de la covariance de la première classe dans l'espace whitened
        S = whitening_mat.T @ covariances[classes[0]] @ whitening_mat
        eigvals_S, eigvecs_S = np.linalg.eigh(S)
        # Tri décroissant
        indices = np.argsort(eigvals_S)[::-1]
        eigvals_S = eigvals_S[indices]
        eigvecs_S = eigvecs_S[:, indices]

        # Reconstruction des filtres spatiaux
        filters = whitening_mat @ eigvecs_S  # de dimension (n_channels, r)
        n_filters = filters.shape[1]
        if self.n_components > n_filters:
            print(
                f"[WARNING] n_components ({self.n_components}) est supérieur au nombre de filtres disponibles ({n_filters}). Réduction de n_components à {n_filters}.")
            self.n_components = n_filters
        # Sélection des filtres extrêmes (les premiers et les derniers)
        n_select = self.n_components // 2
        if self.n_components % 2 == 0:
            selected_filters = np.hstack([filters[:, :n_select], filters[:, -n_select:]])
        else:
            selected_filters = np.hstack([filters[:, :n_select + 1], filters[:, -n_select:]])
        self.filters_ = selected_filters
        return self

    def transform(self, X):
        """
        Applique la transformation CSP aux données EEG.

        Paramètres
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)

        Retourne
        -------
        features : ndarray, shape (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné. Appelez fit avant transform.")

        features = []
        for trial in X:
            # Projection sur les filtres spatiaux
            projected = self.filters_.T @ trial
            # Calcul des variances par composante
            var = np.var(projected, axis=1)
            # Normalisation par la somme totale des variances
            var_norm = var / np.sum(var)
            feature = np.log(var_norm) if self.log else var_norm
            features.append(feature)
        return np.array(features)

    def fit_transform(self, X, y):
        """
        Entraîne le modèle et applique immédiatement la transformation.
        """
        self.fit(X, y)
        return self.transform(X)
