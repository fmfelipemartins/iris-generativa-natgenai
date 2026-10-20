"""Protocolo de refinamento: configuração explícita e seleção apenas no treino."""

from copy import deepcopy  # Isola configurações entre execuções.
import numpy as np  # Operações numéricas determinísticas.
from scipy.stats import (
    wasserstein_distance,
)  # Fidelidade marginal em unidades padronizadas.
from sklearn.metrics import (
    pairwise_distances,
)  # Distâncias para cobertura e proximidade.
from sklearn.ensemble import RandomForestClassifier  # Discriminador fixo de origem.
from sklearn.linear_model import (
    LogisticRegression,
)  # Controle linear de distinguibilidade.
from sklearn.pipeline import (
    make_pipeline,
)  # Ajusta escala apenas no treino discriminativo.
from sklearn.preprocessing import (
    StandardScaler,
)  # Evita vazamento no classificador linear.
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)  # Avaliação estratificada.
from . import config  # Valores atuais preservados como referência.
from .evolution import (
    initialize_populations,
    populations_to_synthetic,
)  # Controle sem evolução.
from .generative import ALL_CONDITIONS  # Interface comum aos seis métodos.

FAMILIES = tuple(ALL_CONDITIONS)  # Ordem estável de execução.
QUALITY_KEYS = (
    "wasserstein",
    "covariance_error",
    "correlation_error",
    "diversity_error",
    "coverage",
    "near_copy_rate",
    "incoherence",
)  # Todos são perdas; menor é melhor.


def explicit_options(condition, generations=40, epochs=30):
    """Materializa parâmetros relevantes antes de executar ou salvar um job."""
    opts = dict(
        pop_size=config.REFINEMENT_POP_SIZE,
        generations=generations,
        show_progress=False,
        record_diagnostics=True,
        selection_policy=config.SELECTION_POLICY,
        fitness_weights=dict(config.FITNESS_WEIGHTS),
    )  # Opções compartilhadas.
    if condition in config.MAIN_CONDITIONS:
        opts.update(
            rmp=config.RMP,
            sbx_eta=config.SBX_ETA,
            sbx_mode="symmetric",
            tournament_size=config.TOURNAMENT_SIZE,
            use_polynomial_mutation=False,
        )  # Referência corrigida.
    else:
        opts.update(
            parent_fraction=config.EDA_PARENT_FRACTION,
            parent_policy="top",
            boundary_policy="clip",
            resample_attempts=5,
            vae_epochs=epochs,
            warm_start=config.VAE_WARM_START,
            refit_every=config.VAE_REFIT_EVERY,
        )  # Ciclo EDA explícito.
        if condition == "vae_eda":
            opts["model_options"] = dict(
                latent=config.VAE_LATENT_DIM,
                hidden=config.VAE_HIDDEN_DIM,
                learning_rate=config.VAE_LEARNING_RATE,
                beta=config.VAE_BETA,
                batch_size=config.VAE_BATCH_SIZE,
                kl_warmup_steps=0,
            )  # Sem promoção automática.
        else:
            opts["model_options"] = dict(
                own_weight=config.GAUSSIAN_OWN_WEIGHT,
                product_weight=config.GAUSSIAN_PRODUCT_WEIGHT,
                include_all=config.GAUSSIAN_INCLUDE_ALL_PRODUCT,
                covariance=config.GAUSSIAN_COVARIANCE,
                regularization=config.GAUSSIAN_REG,
                target_products=False,
            )  # Mistura por tarefa, não EM/GMM.
    return opts  # Snapshot independente dos argumentos padrão das funções.


def candidate_catalog(condition):
    """Até oito variantes predefinidas por família, incluindo a referência."""
    variants = {"reference": {}}  # Controle sempre executado e elegível.
    if condition in config.MAIN_CONDITIONS:
        if condition != "OB-Scan":
            variants.update(
                eta5={"sbx_eta": 5.0}, eta15={"sbx_eta": 15.0}, eta30={"sbx_eta": 30.0}
            )  # Dispersão com ramo corrigido.
        variants.update(
            rmp0={"rmp": 0.0}, rmp01={"rmp": 0.1}, tournament2={"tournament_size": 2}
        )  # Transferência e pressão.
        if condition != "OB-Scan":
            variants["eta15_rmp0"] = dict(
                sbx_eta=15.0, rmp=0.0
            )  # Interação predefinida.
    elif condition == "gaussian_mixture":
        variants.update(
            parent_niches={"parent_policy": "niching"},
            parents075={"parent_fraction": 0.75},
            parents_all={"parent_fraction": 1.0},
            own09={"model_options": {"own_weight": 0.9}},
            own1={"model_options": {"own_weight": 1.0}},
            diagonal={"model_options": {"covariance": "diagonal"}},
            regularized={"model_options": {"regularization": 1e-3}},
        )  # Isola causas antes de combinar.
    elif condition == "gaussian_product":
        variants.update(
            parent_niches={"parent_policy": "niching"},
            target_products={"model_options": {"target_products": True}},
            target02={
                "model_options": {"target_products": True, "product_weight": 0.2}
            },
            no_products={"model_options": {"product_weight": 0.0}},
            products02={"model_options": {"product_weight": 0.2}},
            diagonal={"model_options": {"covariance": "diagonal"}},
            regularized={"model_options": {"regularization": 1e-4}},
        )  # Zero produto preserva o controle uniforme.
    else:
        variants.update(
            parent_niches={"parent_policy": "niching"},
            parents075={"parent_fraction": 0.75},
            all_beta01={"parent_fraction": 1.0, "model_options": {"beta": 0.1}},
            niches_beta01_latent8={
                "parent_policy": "niching",
                "model_options": {"beta": 0.1, "latent": 8},
            },
            all_beta001_latent2={
                "parent_fraction": 1.0,
                "model_options": {"beta": 0.01, "latent": 2},
            },
            niches_warmup={
                "parent_policy": "niching",
                "model_options": {"kl_warmup_steps": 300},
            },
            all_beta01_warmup={
                "parent_fraction": 1.0,
                "model_options": {"beta": 0.1, "kl_warmup_steps": 300},
            },
        )  # Interações que não repetem apenas a ablação anterior.
    return variants  # Grade pequena congelada antes da avaliação.


def options_for(condition, variant, generations=40, epochs=30, overrides=None):
    """Combina o snapshot da referência com sobrescritas explícitas."""
    opts = explicit_options(
        condition, generations, epochs
    )  # Não depende de mutar config global.
    patch = deepcopy(
        candidate_catalog(condition)[variant] if overrides is None else overrides
    )  # Configuração do candidato.
    model_patch = patch.pop(
        "model_options", {}
    )  # Preserva os demais parâmetros do modelo.
    opts.update(patch)  # Sobrescritas no ciclo evolutivo.
    if model_patch:
        opts["model_options"].update(model_patch)  # Mescla apenas campos fornecidos.
    return opts  # Pode ser serializado integralmente no manifesto.


def balanced_sample(points, labels, counts, seed, return_indices=False):
    """Subamostra sem reposição; a população evolutiva permanece em 50 por tarefa."""
    rng = np.random.default_rng(seed)  # RNG exclusivo da avaliação.
    indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == task), size=count, replace=False)
            for task, count in sorted(counts.items())
        ]
    )  # Mesmo orçamento por origem.
    if return_indices:
        return (
            points[indices],
            labels[indices],
            indices,
        )  # Rastreia os reais efetivamente avaliados.
    return points[indices], labels[indices]  # Não altera a população completa.


def quality_metrics(real, real_labels, synthetic, synthetic_labels, context):
    """Métricas por espécie; escalas e limiar de proximidade vêm somente do treino."""
    real_z = context["scaler"].transform(real)  # Avaliação usa transformação do treino.
    synthetic_z = context["scaler"].transform(
        synthetic
    )  # Mesmas unidades para todas as métricas.
    rows = []  # Mantém diagnóstico por tarefa antes da média.
    for task in context["tasks"]:
        a, b = (
            real_z[real_labels == task],
            synthetic_z[synthetic_labels == task],
        )  # Compara dentro da espécie.
        if min(len(a), len(b)) < 2:
            raise ValueError("Métricas exigem ao menos dois pontos por espécie.")
        training = context["real_by_task"][task]  # Referência exclusiva de calibração.
        td = pairwise_distances(training)  # Perfil de proximidade real-real.
        np.fill_diagonal(td, np.inf)  # Exclui a própria observação.
        positive = td.min(1)  # Distâncias ao vizinho mais próximo no treino.
        threshold = max(
            float(np.quantile(positive, 0.05)), 1e-12
        )  # Limiar descritivo; não garantia de privacidade.
        train_distance = pairwise_distances(b, training).min(
            1
        )  # Proximidade aos reais usados para gerar.
        da, db = (
            pairwise_distances(a),
            pairwise_distances(b),
        )  # Diversidade intratarefa.
        ratio = float(
            db[np.triu_indices(len(b), 1)].mean()
            / max(da[np.triu_indices(len(a), 1)].mean(), 1e-12)
        )  # Relativa ao real reservado.
        covariance_a, covariance_b = (
            np.cov(a, rowvar=False),
            np.cov(b, rowvar=False),
        )  # Relações conjuntas.

        def correlation(cov):
            scales = np.sqrt(
                np.maximum(np.diag(cov), 1e-12)
            )  # Estável para candidatos constantes.
            return cov / np.outer(scales, scales)  # Evita NaN de variância nula.

        nearest_tasks = np.array(context["tasks"])[
            np.column_stack(
                [
                    pairwise_distances(b, context["real_by_task"][t]).min(1)
                    for t in context["tasks"]
                ]
            ).argmin(1)
        ]  # Coerência descritiva baseada só no treino.
        rows.append(
            dict(
                task=task,
                wasserstein=float(
                    np.mean(
                        [
                            wasserstein_distance(a[:, j], b[:, j])
                            for j in range(a.shape[1])
                        ]
                    )
                ),
                covariance_error=float(
                    np.linalg.norm(covariance_a - covariance_b, "fro")
                    / max(np.linalg.norm(covariance_a, "fro"), 1e-12)
                ),
                correlation_error=float(
                    np.linalg.norm(
                        correlation(covariance_a) - correlation(covariance_b), "fro"
                    )
                    / a.shape[1]
                ),
                diversity_ratio=ratio,
                diversity_error=float(abs(np.log(max(ratio, 1e-12)))),
                coverage=float(pairwise_distances(a, b).min(1).mean()),
                near_copy_rate=float(np.mean(train_distance < threshold)),
                exact_duplicate_rate=float(
                    np.mean(
                        np.any(
                            np.all(b[:, None, :] == training[None, :, :], axis=2),
                            axis=1,
                        )
                    )
                ),
                incoherence=float(np.mean(nearest_tasks != task)),
            )
        )  # Cada métrica conserva sua unidade.
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "task"
    }  # Média igual entre espécies.
    return aggregate, rows  # Não transforma menor acurácia em objetivo único.


def calibration(context, counts, seed, repetitions=20):
    """Piso e tolerâncias descritivos via duas subamostras reais disjuntas do treino."""
    rng = np.random.default_rng(seed)  # Não consome o RNG de geração.
    measurements = []  # Variabilidade real-real com tamanho comparável ao teste.
    for _ in range(repetitions):
        a, b, labels = [], [], []  # Amostras balanceadas por espécie.
        for task in context["tasks"]:
            pool = context["real_by_task"][task]  # Nenhum dado externo entra aqui.
            count = min(
                counts[task], len(pool) // 2
            )  # Subamostras obrigatoriamente disjuntas.
            indices = rng.choice(
                len(pool), 2 * count, replace=False
            )  # Sem autocópia entre as duas amostras.
            a.append(pool[indices[:count]])  # Primeiro grupo real.
            b.append(pool[indices[count:]])  # Segundo grupo real.
            labels.extend([task] * count)  # Identidade das tarefas.
        a, b = (
            context["scaler"].inverse_transform(np.vstack(a)),
            context["scaler"].inverse_transform(np.vstack(b)),
        )  # Interface em unidades originais.
        measures, _ = quality_metrics(
            a, np.asarray(labels), b, np.asarray(labels), context
        )  # Distâncias de cópia não calibradas com membros do treino.
        measurements.append(measures)  # Guarda perdas real-real.
    scale = {
        key: max(float(np.median([m[key] for m in measurements])), 0.02)
        for key in QUALITY_KEYS
    }  # Evita divisão por zero.
    scale.update(
        near_copy_rate=0.05, incoherence=0.05
    )  # Tolerância fixa de cinco pontos percentuais nestas taxas.
    return scale  # Heurística declarada, não teste estatístico.


def quality_loss(metrics, scale):
    """Agregação predefinida para triagem interna; não substitui a tabela de métricas."""
    return float(
        np.mean([metrics[key] / scale[key] for key in QUALITY_KEYS])
    )  # Peso igual após normalização real-real.


def discriminators(real, labels, synthetic, synthetic_labels, seed, trees=300):
    """Distingue origens com folds próprios; não escolhe hiperparâmetros do gerador."""
    points = np.vstack((real, synthetic))  # Origens balanceadas por espécie.
    origin = np.r_[
        np.ones(len(real), dtype=int), np.zeros(len(synthetic), dtype=int)
    ]  # 1 real, 0 sintético.
    tasks = np.r_[labels, synthetic_labels]  # Estratificação conjunta.
    strata = np.array(
        [f"{o}_{t}" for o, t in zip(origin, tasks)]
    )  # Preserva origem e espécie.
    splits = list(
        StratifiedKFold(3, shuffle=True, random_state=seed).split(points, strata)
    )  # Separado dos folds de seleção.
    output = {}  # Mantém RF e modelo linear como controles distintos.
    models = {
        "rf": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
    }  # Configurações fixas entre variantes.
    for name, model in models.items():
        scores = cross_validate(
            model,
            points,
            origin,
            cv=splits,
            scoring={"accuracy": "accuracy", "auc": "roc_auc"},
            n_jobs=1,
        )  # Escala da regressão ajustada por fold.
        output.update(
            {
                f"{name}_{metric}": float(scores[f"test_{metric}"].mean())
                for metric in ("accuracy", "auc")
            }
        )  # Médias descritivas.
    return output  # Menor distinguibilidade não é aprovação isolada.


def initialization(context, seed):
    """Controle da população inicial, com os mesmos 50 indivíduos por espécie."""
    populations = initialize_populations(
        context, config.REFINEMENT_POP_SIZE, seed
    )  # Mesmo ponto de partida do gerador.
    z, x, labels = populations_to_synthetic(
        populations, context
    )  # Unidades originais e padronizadas.
    return dict(
        synthetic_z=z, synthetic_x=x, synthetic_species=labels
    )  # Sem evolução nem ajuste de modelo.
