# Metodologias e decisões da iteração 2.3

Referência principal: [Evolutionary Computation as Natural Generative AI](https://arxiv.org/html/2510.08590v1), especialmente a discussão de moderação e os experimentos artificiais das seções VI-A/VI-B. A tradução fornecida pelo projeto foi utilizada como apoio de leitura.

## Correspondência com o artigo

| Elemento | Implementação nesta iteração | Alcance |
|---|---|---|
| Moderação da seleção | `global`, `task`, `task_niching` | Controle experimental inspirado na proteção de tarefas; não implementa todo o MFEA. Global fica no laboratório 2D. |
| VAE-EDA | VAE condicional, uma camada oculta no encoder/decoder, latente 32; reajuste em pais selecionados | Arquitetura inspirada no artigo; largura 64, Adam, β=1 e orçamento são escolhas explícitas da aplicação. |
| Mistura gaussiana | Média/covariância por tarefa e amostragem de mistura | Não usa EM nem escolhe automaticamente o número de componentes. |
| Mistura + produto | Componentes individuais e produtos gaussianos normalizados | Pares por padrão; produto de todas as tarefas opcional. |
| Cenários 2D | Anel/espiral e Gaussianas anisotrópicas | Geometria é reconstrução ilustrativa. Parâmetros de origem gaussianos seguem o artigo; modelos são estimados nos pais amostrados. |
| OB-Scan-KDE e híbrido | Operadores anteriores preservados | São discutidos no artigo; não são apresentados como invenção original do projeto. |
| Iris e avaliações | Fitness P/N/C, nichos KMeans e testes discriminativos | Adaptações do projeto; não equivalem às avaliações físicas e semânticas 3D. |

## Gaussianas

Para cada tarefa, estimamos média e covariância da fração superior da população por fitness. A covariância recebe regularização diagonal de `1e-6`. `full` usa correlações entre atributos; `diagonal` é uma opção de ablação.

O produto normalizado de gaussianas usa `Σ = (Σ₁⁻¹ + … + Σₖ⁻¹)⁻¹` e `μ = Σ(Σ₁⁻¹μ₁ + … + Σₖ⁻¹μₖ)`, implementados com sistemas lineares. Cada produto é normalizado separadamente antes de entrar na mistura. Assim, o peso 0,40 especifica massa de probabilidade, não amplitude de um produto de densidades não normalizado.

Na Iris, a mistura por tarefa atribui 0,70 à tarefa-alvo e divide 0,30 entre as demais. Na variante com produtos, as três gaussianas individuais compartilham 0,60, e os três pares compartilham 0,40. Com o produto triplo habilitado, os quatro produtos dividem essa mesma massa. Essa extensão para três tarefas é uma decisão da aplicação. A distribuição proposta pela variante produto é a mesma para todas as tarefas; o fitness e a seleção de sobreviventes dependem da tarefa-alvo.

No laboratório 2D, as fontes são A: média (2,8), covariância diag(0,2;1); B: média (8,2), covariância diag(1;0,2). As misturas por alvo são 0,7A+0,3B e 0,5A+0,5B; a variante produto é 0,3A+0,3B+0,4P, onde P é o produto normalizado dos modelos estimados.

## Ciclo EDA e VAE

1. Inicializar a população por reamostragem dos reais da tarefa mais jitter gaussiano.
2. Selecionar os melhores 50% de cada tarefa pelo fitness para ajustar o modelo.
3. Reajustar gaussianas ou treinar o VAE; amostrar 50 candidatos por tarefa.
4. Aplicar limites do contexto, reunir pais e filhos e selecionar sobreviventes.
5. Repetir pelo orçamento de gerações.

O VAE recebe condição one-hot, usa reparametrização e decodifica latentes normais. O objetivo soma erro quadrático de reconstrução por atributo e divergência KL por dimensão latente. O treinamento é em CPU, com gerador aleatório privado. Pesos e estado do Adam continuam apenas entre gerações de uma mesma execução quando `VAE_WARM_START=True`. `VAE_REFIT_EVERY` controla a frequência; desativar warm start reinicia o modelo nas gerações de ajuste. Cada chamada do gerador cria estado independente, inclusive nos folds.

## Seleção e controles

Na Iris, as duas políticas preservam o orçamento de cada tarefa: ranking por fitness (`task`) e preservação proporcional dos nichos seguida de preenchimento por fitness (`task_niching`). A seleção parental do EDA permanece baseada na fração superior por tarefa em ambas; a ablação atua na sobrevivência.

O controle `global` classifica todos os candidatos por fitness na tarefa atribuída e pode eliminar tarefas. O laboratório 2D acompanha contagens de tarefas e fitness por geração. O controle contrafactual aplica as três políticas ao mesmo conjunto de candidatos de cada trajetória e conta sobreviventes, descendentes e descendentes de cruzamentos entre tarefas. As trajetórias posteriores diferem justamente pela sobrevivência anterior. As escalas do fitness são calibradas por tarefa; ranking global é um controle dessa implementação, não uma medida universal de qualidade entre tarefas.

## Avaliação e interpretação

Mesmas seeds e gerações tornam o protocolo pareado, mas não igualam custo computacional. Tempo e número de avaliações de fitness são registrados; treinamento do VAE acrescenta custo que não aparece nessa contagem. Os benchmarks de amostragem não são uma busca evolutiva completa e usam limites diferentes: crossovers têm caixa derivada dos pais, enquanto Gaussianas/VAE não são truncados no laboratório 2D. Na Iris todos os geradores novos são limitados à caixa do contexto.

Cross-fitting refaz padronização, calibração e modelos usando exclusivamente o treinamento externo. O discriminador interno usa somente reais reservados e sintéticos gerados daquele treinamento. O escore mede distinguibilidade nesse protocolo; não comprova equivalência das distribuições. A fração de pontos fora do casco convexo é descritiva, não evidência suficiente de criatividade ou generalização OOD.
