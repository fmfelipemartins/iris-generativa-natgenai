"""Configuração experimental; novos valores são escolhas explícitas da aplicação."""

SEED = 42  # Seed de referência da inicialização e evolução.
CV_SEED = 2026  # Seed da divisão dos dados e dos discriminadores.
POP_SIZE = 50  # Indivíduos mantidos por espécie no experimento Iris.
GENERATIONS = 40  # Gerações de cada execução completa.
RMP = 0.20  # Probabilidade de escolher pais de tarefas diferentes.
TOURNAMENT_SIZE = 3  # Candidatos sorteados para escolher cada pai.
SBX_ETA = 50.0  # Índice de distribuição do crossover SBX.
POLY_ETA = 20.0  # Índice de distribuição da mutação polinomial.
POLY_MUTATION_PROB = 0.20  # Probabilidade de mutação por gene no laboratório separado.
INITIAL_JITTER = 0.08  # Desvio do ruído inicial em unidades padronizadas.
N_NICHES = 5  # Clusters KMeans fixos por tarefa.
K_NEIGHBORS = 5  # Vizinhos utilizados na calibração e fitness.
FITNESS_WEIGHTS = {
    "plausibility": 0.50,  # Peso da proximidade local aos reais da tarefa.
    "novelty": 0.25,  # Peso da proximidade à distância-alvo de novidade.
    "coherence": 0.25,  # Peso da compatibilidade relativa com a tarefa.
}
MAIN_CONDITIONS = {
    "SBX": 1.0,  # Probabilidade de SBX: crossover exclusivamente SBX.
    "OB-Scan": 0.0,  # Probabilidade de SBX: crossover exclusivamente OB-Scan-KDE.
    "Híbrido 50/50": 0.5,  # Probabilidade por descendente, sem cota fixa.
}
GENERATIVE_METHODS = (
    "gaussian_mixture",  # Mistura explícita de distribuições por tarefa.
    "gaussian_product",  # Mistura de componentes individuais e produtos normalizados.
    "vae_eda",  # VAE condicional atualizado na população selecionada.
)
SELECTION_POLICY = "task_niching"  # Referência: cotas por tarefa e por nicho.
GAUSSIAN_REG = 1e-6  # Regularização diagonal das covariâncias em unidades ao quadrado.
GAUSSIAN_COVARIANCE = "full"  # Covariância completa; alternativa: diagonal.
GAUSSIAN_OWN_WEIGHT = 0.70  # Massa da tarefa-alvo na mistura Iris; restante nas demais.
GAUSSIAN_PRODUCT_WEIGHT = 0.40  # Massa total dos componentes de produto.
GAUSSIAN_INCLUDE_ALL_PRODUCT = False  # Incluir produto conjunto além dos pares.
EDA_PARENT_FRACTION = 0.50  # Fração superior por tarefa usada no ajuste dos modelos.
VAE_LATENT_DIM = 32  # Dimensão latente da arquitetura de referência do artigo.
VAE_HIDDEN_DIM = 64  # Largura da única camada oculta; escolha da aplicação.
VAE_EPOCHS = 30  # Épocas por atualização do VAE no perfil completo.
VAE_BATCH_SIZE = 32  # Registros por lote de treinamento.
VAE_LEARNING_RATE = 1e-3  # Taxa de aprendizado do Adam.
VAE_BETA = 1.0  # Peso do termo KL no objetivo do VAE.
VAE_WARM_START = True  # Reaproveitar pesos e Adam somente dentro da mesma execução.
VAE_REFIT_EVERY = 1  # Intervalo de gerações entre ajustes do VAE.
BENCHMARK_SIZE = 200  # Pais por tarefa nos cenários bidimensionais completos.
BENCHMARK_GENERATIONS = 20  # Gerações do controle de seleção em 2D.
SHOW_PROGRESS = True  # Exibir progresso sem consumir números aleatórios.
EXECUTION_PROFILE = "quick"  # quick: revisão funcional; full: protocolo completo.
QUICK_GENERATIONS = 3  # Gerações da validação funcional reduzida.
QUICK_VAE_EPOCHS = 3  # Épocas por ajuste no perfil reduzido.
QUICK_BENCHMARK_SIZE = 40  # Pais por tarefa nos cenários reduzidos.
ROBUSTNESS_SEEDS = tuple(range(20))  # Seeds pareadas do protocolo completo.
RUN_ROBUSTNESS = True  # Habilitar protocolo de robustez.
RUN_SENSITIVITY = True  # Habilitar sensibilidade e ablações existentes.
RUN_CROSSFIT = True  # Habilitar protocolo com reais reservados.
RUN_POLYNOMIAL_LAB = True  # Habilitar laboratório separado de mutação.
RUN_EXTENDED = True  # Habilitar as três famílias de geradores adicionais.
RUN_BENCHMARKS = True  # Habilitar cenários artificiais bidimensionais.
RUN_SELECTION_STUDY = True  # Habilitar comparação explícita das políticas de seleção.
