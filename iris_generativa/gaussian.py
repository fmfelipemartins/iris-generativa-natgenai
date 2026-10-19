"""Misturas por tarefa e produtos NORMALIZADOS de Gaussianas (não EM/GMM)."""

from itertools import combinations

import numpy as np

from .config import GAUSSIAN_REG


def fit_gaussian(points, regularization=GAUSSIAN_REG, covariance="full"):
    """Estima média/covariância; regulariza também populações constantes."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Ajuste requer pelo menos dois vetores finitos.")
    if regularization <= 0 or covariance not in {"full", "diagonal"}:
        raise ValueError(
            "Regularização positiva e covariância full/diagonal obrigatórias."
        )
    cov = np.atleast_2d(np.cov(points, rowvar=False))
    if covariance == "diagonal":
        cov = np.diag(np.diag(cov))
    cov = (cov + cov.T) / 2 + regularization * np.eye(points.shape[1])
    return points.mean(axis=0), cov


def gaussian_product(models):
    """Parâmetros da densidade proporcional ao produto, normalizada para integrar 1."""
    if len(models) < 2:
        raise ValueError("O produto requer pelo menos duas Gaussianas.")
    dimension = len(models[0][0])
    identity = np.eye(dimension)
    precisions = [np.linalg.solve(cov, identity) for _, cov in models]
    precision = sum(precisions)
    covariance = np.linalg.solve(precision, identity)
    mean = np.linalg.solve(
        precision, sum(p @ mu for p, (mu, _) in zip(precisions, models))
    )
    return mean, (covariance + covariance.T) / 2


def sample_components(models, weights, size, rng):
    """Amostra componentes e devolve sua origem para auditoria."""
    weights = np.asarray(weights, dtype=float)
    if (
        len(weights) != len(models)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Pesos finitos, não negativos e com soma 1 obrigatórios.")
    if size < 0:
        raise ValueError("Tamanho negativo.")
    indices = rng.choice(len(models), size=size, p=weights)
    points = np.empty((size, len(models[0][0])))
    for index, (mean, cov) in enumerate(models):
        mask = indices == index
        points[mask] = rng.multivariate_normal(
            mean, cov, size=int(mask.sum()), check_valid="raise"
        )
    return points, indices


class GaussianGenerator:
    """Uma Gaussiana por tarefa; pares e produto conjunto são componentes opcionais."""

    def __init__(
        self,
        method="gaussian_mixture",
        own_weight=0.7,
        product_weight=0.4,
        include_all=False,
        covariance="full",
        regularization=GAUSSIAN_REG,
    ):
        if method not in {"gaussian_mixture", "gaussian_product"}:
            raise ValueError("Método gaussiano desconhecido.")
        if not 0 <= own_weight <= 1 or not 0 <= product_weight <= 1:
            raise ValueError("Massas devem pertencer a [0, 1].")
        self.method, self.own_weight = method, own_weight
        self.product_weight, self.include_all = product_weight, include_all
        self.covariance, self.regularization = covariance, regularization

    def fit(self, populations):
        self.tasks = sorted(populations)
        self.models = [
            fit_gaussian(populations[t], self.regularization, self.covariance)
            for t in self.tasks
        ]
        self.components = [(t,) for t in self.tasks]
        if self.method == "gaussian_product" and len(self.tasks) > 1:
            groups = list(combinations(range(len(self.tasks)), 2))
            if self.include_all and len(self.tasks) > 2:
                groups.append(tuple(range(len(self.tasks))))
            base = list(self.models)
            for group in groups:
                self.models.append(gaussian_product([base[i] for i in group]))
                self.components.append(tuple(self.tasks[i] for i in group))
        return self

    def sample(self, task, size, rng):
        n_tasks = len(self.tasks)
        if task not in self.tasks:
            raise ValueError("Tarefa desconhecida.")
        n_products = len(self.models) - n_tasks
        if n_products:
            # Adaptação Iris: massa individual uniforme; produtos uniformes entre pares.
            weights = [(1 - self.product_weight) / n_tasks] * n_tasks
            weights += [self.product_weight / n_products] * n_products
        elif n_tasks == 1:
            weights = [1.0]
        else:
            weights = [(1 - self.own_weight) / (n_tasks - 1)] * n_tasks
            weights[self.tasks.index(task)] = self.own_weight
        points, origins = sample_components(self.models, weights, size, rng)
        return points, [self.components[i] for i in origins]
