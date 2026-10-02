"""Configuração experimental da Iris Generativa v2.2."""

SEED = 42
CV_SEED = 2026

POP_SIZE = 50
GENERATIONS = 40
RMP = 0.20

SBX_ETA = 50.0
POLY_ETA = 20.0
POLY_MUTATION_PROB = 0.20

INITIAL_JITTER = 0.08
N_NICHES = 5
K_NEIGHBORS = 5

FITNESS_WEIGHTS = {
    "plausibility": 0.50,
    "novelty": 0.25,
    "coherence": 0.25,
}

MAIN_CONDITIONS = {
    "SBX": 1.0,
    "OB-Scan": 0.0,
    "Híbrido 50/50": 0.5,
}

RUN_ROBUSTNESS = True
RUN_SENSITIVITY = True
RUN_CROSSFIT = True
RUN_POLYNOMIAL_LAB = True
