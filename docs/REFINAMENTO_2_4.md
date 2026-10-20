# Refinamento 2.4 — protocolo e interpretação

Esta iteração mantém seis famílias gerativas, cinco nichos fixos e híbrido probabilístico. Não promove hiperparâmetros automaticamente nem altera os padrões da versão 2.3.1. As alternativas são configurações explícitas dos métodos existentes.

## Alterações implementadas

- Fitness configurável por execução, sem mutar o contexto compartilhado.
- Torneio configurável; diagnósticos e geração de origem dos sobreviventes nos crossovers, sem mudar o RNG nem o mecanismo de seleção.
- EDAs com seleção parental `top` ou `niching`. A segunda reutiliza os cinco nichos e suas cotas, sem clusterização dinâmica.
- Produtos gaussianos com `target_products=True`: massa individual na espécie-alvo e massa de produtos dividida apenas entre componentes que incluem essa espécie. A variante uniforme anterior permanece como controle.
- VAE com rampa linear opcional de KL, medida em atualizações do otimizador e contínua entre gerações quando há warm start. Histórico registra beta efetivo e atualizações.
- Política opcional de reamostragem de candidatos fora dos limites: até cinco tentativas, seguida de clipping residual. Registra o custo extra; não altera silenciosamente a versão de referência.
- Executor com triagem, confirmação, refinamentos e avaliação externa, checkpoints atômicos, lock exclusivo e verificação de hashes do código, configuração, dados, Python, plataforma e bibliotecas na retomada.

## Busca predefinida

A primeira etapa testa no máximo oito configurações por família, incluindo referência. O catálogo completo está em `candidate_catalog`, no módulo `refinement.py`, e é serializado no manifesto de cada execução.

| Família | Alternativas principais |
|---|---|
| SBX/híbrido | η=5/15/30/50; RMP=0/0,1/0,2; torneio=2/3; uma interação η=15/RMP=0 |
| OB-Scan | RMP=0/0,1/0,2 e torneio=2/3 |
| Mistura | Pais por nichos; fração parental=0,75/1; peso próprio=0,9/1; diagonal; regularização=1e-3 |
| Produtos | Pais por nichos; componentes condicionados; massa de produtos=0/0,2/0,4; diagonal; regularização=1e-4 |
| VAE | Nichos; fração=0,75; interações fração/β/latente; β=0,01/0,1/1; latente=2/8/32; warmup de 300 atualizações |

Não se executa o produto cartesiano desses valores. Na segunda etapa, o melhor candidato elegível recebe variantes de pesos do fitness (0,4/0,35/0,25 e 0,4/0,25/0,35), e o VAE também testa taxa 3e-4 e ajuste a cada duas gerações com o dobro de épocas por ajuste. Se o clipping médio interno ultrapassar 5%, inclui-se a reamostragem. A comparação de refit aproxima o número total de épocas em gerações pares; não garante o mesmo estado do Adam ou as mesmas amostras de treinamento.

## Separação de dados e orçamento

No perfil `full`, são cinco folds externos. Em cada treino externo, duas subdivisões internas fazem toda a seleção; scaler, calibração do fitness, nichos e modelos são ajustados somente no treino de cada subdivisão. O teste externo nunca escolhe a variante.

- Triagem: dez gerações, seeds 0/1/2; VAE com 30 épocas por ajuste.
- Confirmação: referência e até duas variantes, 40 gerações, seeds 3–7.
- Refinamentos: mesmo orçamento da confirmação; todos passam pelas mesmas guardas.
- Teste externo: referência, variante escolhida e inicialização sem evolução, seeds 20–24. Se a referência for selecionada, não é executada duas vezes.

A evolução mantém 50 indivíduos por espécie, também nos folds. A avaliação externa usa dez sintéticos por espécie, sem reposição, para balancear os 30 reais reservados. O RNG dessa amostra é independente da geração. Nas subdivisões internas, o tamanho avaliado é limitado à metade do treino por espécie, permitindo calibrar a variabilidade com duas subamostras reais disjuntas de tamanho igual. Os índices reais efetivamente avaliados ficam registrados em cada job.

O perfil `quick` reduz folds, seeds, gerações, épocas e árvores; verifica apenas funcionamento. Não prova ganho científico. A fase `search`, aplicada a toda a Iris, é exploração interna e não substitui `nested`.

## Métricas e regra de seleção

A escolha não usa acurácia da Random Forest como objetivo. São sete perdas por espécie: Wasserstein marginal; erro relativo de covariância; erro de correlação; valor absoluto do log da razão de diversidade; distância dos reais reservados aos sintéticos (cobertura); taxa de proximidade excessiva ao treino; incoerência por vizinho mais próximo do treino.

O limiar de proximidade é o percentil 5 das distâncias reais ao vizinho mais próximo no treino, excluindo a diagonal. É um indicador descritivo, não garantia de privacidade. Duplicação exata é registrada separadamente. As correlações usam tratamento explícito para variâncias nulas.

As perdas são normalizadas por escalas real-real calculadas com 20 repetições exclusivamente no treino. Para proximidade e incoerência, a escala é 0,05. A média normalizada tem pesos iguais. O critério da busca combina 50% da perda dos novos candidatos com 50% da perda dos sobreviventes. Assim, reter fundadores não mascara, sozinho, candidatos concentrados.

Uma variante precisa reduzir a perda interna em pelo menos 5%, sem deteriorar cada dimensão acima de 25% de sua escala real-real; a margem das duas taxas é de cinco pontos percentuais. As guardas também são verificadas por espécie e por estágio. Não se aceita aumento da taxa média de duplicação exata nos sobreviventes. Essas margens são escolhas operacionais predefinidas, não significância estatística.

A inicialização não possui candidatos: compare sua `survivor_loss` com a `survivor_loss` dos métodos. A coluna `loss` dos métodos mistura candidatos e sobreviventes e não deve ser comparada diretamente com a perda simples da inicialização.

RF de 300 árvores e regressão logística avaliam as saídas externas e os controles, com CV discriminativa própria de três folds. Esses folds não são a camada de seleção do gerador. A escala da regressão logística é ajustada dentro de cada fold discriminativo.

## Incerteza e limites

As diferenças externas são pareadas por condição, fold e seed. O intervalo bootstrap agrega primeiro os folds por seed e reamostra seeds; quantifica apenas aleatoriedade algorítmica, condicionada à Iris e às divisões fixadas. Cinco seeds não sustentam uma estimativa precisa de incerteza populacional. Os folds compartilham treino e não são tratados como observações independentes.

A Iris já foi explorada em iterações anteriores. A separação aninhada evita novo vazamento programático na seleção, mas não transforma o dataset em uma amostra inédita. Os resultados continuam sendo evidência exploratória para esta base, sem demonstrar generalização para outra aplicação.

A soma de tempo das execuções inclui geração e avaliação; `elapsed_seconds` identifica o tempo do gerador. Execuções paralelas não equivalem a benchmark isolado de tempo. Também se registram avaliações de fitness, propostas adicionais e atualizações do otimizador. Mesmo número de gerações não iguala custo entre famílias.

## Executar

Na raiz do projeto, mantendo o ambiente atual:

```powershell
python -m pytest -q
python scripts/refine_experiments.py --profile quick --phase controls --output results/refinement_local_quick --workers 1
python scripts/refine_experiments.py --profile full --phase nested --output results/refinement_local_full --workers 4
```

Para retomar a mesma fase, adicione `--resume` sem alterar código, configuração ou ambiente. Para nova tentativa, use outra pasta. Mudança de ambiente exige nova execução; a verificação não deve ser desabilitada. Um lock remanescente exige confirmar que não há executor ativo antes de removê-lo.

`manifest.json` registra o protocolo; `jobs/` guarda dados, índices e diagnósticos; `runs.csv` e `summary.csv` consolidam resultados; `external.csv`, `paired_external.csv` e `conditional_uncertainty.csv` isolam a avaliação externa. `COMPLETE.json` só é escrito ao concluir toda a fase solicitada. Interromper não equivale a completar a avaliação.

O notebook principal mantém a suíte anterior para comparação e acrescenta uma seção própria para o refinamento. O cross-fitting histórico, com população menor, não deve ser misturado com os números do protocolo novo.
