# Revisão metodológica 2.3.1 — 19/10/2026

Esta revisão preserva a execução completa anterior, corrige uma assimetria confirmada no SBX e torna observável o que o VAE produz antes da seleção. Os hiperparâmetros de treinamento do VAE permanecem nos valores de referência. Nichos dinâmicos e divisão exata de 50/50 do híbrido continuam fora do escopo.

## Entregas do plano

| Etapa | Implementação |
|---|---|
| Preservar a referência `full` | Notebook enviado pelo usuário, sem alteração de bytes, em `results/baseline_full_20261019`, com SHA-256 e metadados. |
| Diagnosticar assimetria SBX | Pais controlados, ordem invertida, η=5/50 e 10.000 amostras por combinação; resultados em `sbx_controlled.csv`. |
| Corrigir se confirmada | Ramos equiprováveis por gene e limites por atributo; modo `legacy_lower` para auditoria histórica. |
| Instrumentar VAE | Estágios antes/depois da seleção, diversidade intratarefa, fundadores, KL, unidades ativas e resposta do decoder. |
| Comparar hiperparâmetros pontualmente | Quatro variantes predefinidas, uma alteração por vez, seeds 0/1/2, orçamento completo. |
| Revalidar protocolos afetados | Comparações pareadas SBX/híbrido, cross-fitting, sensibilidade, seleção, mutação polinomial e cenários 2D. |
| Consolidar documentação e notebook | Pipeline revisada, diagnósticos no notebook e exportação CSV por execução; atualização no mesmo PR. |

## SBX: causa e correção

Na instalação PyGAD 3.7.0 examinada, `sbx_crossover` ordena cada par de genes em `y1 <= y2` e devolve apenas `(y1+y2-beta_q*(y2-y1))/2`. Solicitar dois filhos repete o ramo inferior; inverter os pais não muda essa ordenação. A documentação oficial mais recente de [operadores PyGAD](https://pygad.readthedocs.io/en/latest/utils.html) descreve uma correção de simetria, mas isso não altera o código da versão 3.7.0 já instalada.

O helper local mantém a fórmula de espalhamento limitado dessa versão e sorteia os sinais −/+ independentemente por gene, com probabilidade 0,5. Também utiliza os limites de cada atributo, em vez da caixa escalar do motor legado. Assim, a comparação `legacy_lower` versus `symmetric` incorpora **essas duas diferenças**, além da mudança correspondente no consumo de RNG; não isola exclusivamente o sinal do ramo. Não é uma implementação presumidamente idêntica a todas as variantes de SBX publicadas.

No controle com pais (−1,2) e (1,−2), ambos os genes do legado ficaram abaixo do ponto médio em todas as 10.000 amostras, em cada η. No modo corrigido, as proporções acima do ponto médio foram 50,06% e 50,24%. A inversão da ordem parental produziu as mesmas amostras para a mesma seed. Testes cobrem ainda igualdade dos pais, limites por gene, reprodutibilidade e combinações dos sinais.

Resultados agregados da comparação completa constam em `results/review_full_20261019/sbx_summary.csv`; os valores por seed estão em `sbx_paired.csv`. Há 20 seeds × duas condições × dois modos, com 40 gerações, RF de 300 árvores e CV de cinco folds × cinco repetições. O pareamento usa a mesma seed por condição/modo; não garante que os mesmos números aleatórios sejam usados depois da mudança do operador. O critério para corrigir o defeito é a simetria esperada, não obter menor acurácia do discriminador.

| Condição | Modo | Acurácia RF média ± DP | AUC média ± DP |
|---|---|---:|---:|
| Híbrido 50/50 | legacy_lower | 0.8751 ± 0.0131 | 0.9388 ± 0.0073 |
| Híbrido 50/50 | symmetric | 0.9241 ± 0.0143 | 0.9639 ± 0.0083 |
| SBX | legacy_lower | 0.8610 ± 0.0138 | 0.9373 ± 0.0074 |
| SBX | symmetric | 0.9292 ± 0.0133 | 0.9662 ± 0.0069 |

DP calculado entre as 20 seeds, não entre folds. Estes escores diretos usam os reais que também orientam o gerador e não substituem o cross-fitting.

## VAE: candidatos e população final contam histórias diferentes

Médias das três seeds no protocolo discriminativo direto, com os reais usados como referência pelo gerador. Esses escores são **exploratórios**, não avaliação externa nem critério suficiente para selecionar um modelo.

| Variante | Acurácia RF: candidatos da última geração | Acurácia RF: sobreviventes | Fundadores ainda presentes | Diversidade dos candidatos / reais |
|---|---:|---:|---:|---:|
| Referência | 0,9844 | 0,7622 | 43,56% | 0,0694 |
| β=0,1 | 0,9722 | 0,8067 | 31,33% | 0,0853 |
| Latente=8 | 0,9860 | 0,7442 | 46,22% | 0,0515 |
| Fração parental=1,0 | 0,9551 | 0,8173 | 20,22% | 0,1788 |

A diversidade é a média das razões intratarefa, na geração 40, entre as três tarefas e as três seeds. Os candidatos são avaliados após clipping, antes da sobrevivência. `founder_fraction` identifica indivíduos da inicialização ainda presentes; os demais sobreviventes podem vir de qualquer geração anterior, não apenas da última.

Na referência, a sonda final do encoder encontrou KL médio 0,000291 e nenhuma unidade ativa pelo limiar de variância 0,01. A variância do decoder ao variar z foi 0,005408, enquanto o efeito médio de trocar a condição mantendo z foi 8,715075, em unidades padronizadas ao quadrado. Esses resultados, junto à baixa diversidade dos candidatos, são compatíveis com forte dependência da condição e pouco uso do latente nesta configuração. As duas métricas do decoder têm definições distintas; sua razão não é um teste estatístico de importância.

Com β=0,1, o número médio de unidades ativas subiu para 4,22, mas a diversidade dos candidatos continuou em apenas 8,53% da real. Usar todos os pais aumentou essa razão para 17,88%, ainda baixa. Reduzir o latente melhorou a acurácia da população final, mas diminuiu a diversidade dos candidatos e manteve mais fundadores. Portanto, essa melhoria aparente não justifica escolher latente=8.

**Decisão:** manter β=1, latente=32 e fração parental=0,5. Não há evidência suficiente nesta exploração para declarar resolvida a concentração do VAE ou promover uma variante a novo padrão. A análise sugere investigar futuramente a interação entre concentração parental, objetivo de treinamento e uso do latente. Uma queda de reconstrução ao longo das gerações também pode refletir mudança dos pais de treinamento; não demonstra, isoladamente, melhoria em uma distribuição fixa.

## Validação cruzada e controles

O cross-fitting corrigido usa cinco folds externos, três internos, 40 gerações e RF de 300 árvores, sem usar os reais reservados no ajuste dos geradores. Acurácia/AUC médias: SBX **0,7700/0,8293**; híbrido **0,7500/0,8427**. O notebook anterior registrava 0,7300/0,8540 e 0,7400/0,8453, respectivamente, em outro ambiente. Essa comparação histórica é descritiva: para inferir efeito isolado do operador, usar os modos pareados no mesmo ambiente. Não foi refeito cross-fitting dos quatro métodos não afetados pela correção.

Os cenários 2D foram repetidos com 200 pais por tarefa e 30 épocas VAE. As figuras mostram os descendentes SBX distribuídos nos ramos combinados, sem a concentração unilateral anterior. Isso não elimina todas as concentrações induzidas pelo fitness ou pelos próprios dados. O VAE ainda concentra a geometria anel/espiral nesse orçamento.

O controle de seleção em 2D, com 20 gerações, terminou com 1/399 indivíduos na política global e 200/200 nas duas políticas com cotas. As contagens dependem do operador e da trajetória; o resultado anterior 400/0 é mantido somente como referência histórica. Sensibilidade de η/RMP, ablação de sobrevivência na Iris e SBX com mutação polinomial foram novamente executados com o operador corrigido e o orçamento `full`.

## Rastreabilidade e reprodução

A execução anterior veio do ambiente Windows/Python 3.14.7 do usuário. Esta revisão foi executada em Linux/Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 e PyGAD 3.7.0, com VAE em CPU. Os metadados de cada fase registram as versões instaladas; durante esta sessão a distribuição Torch passou de `2.14.1+cpu` para `2.14.1`, sem mudança do código de treinamento. A fase VAE foi concluída no processo que carregou a versão CPU; após a renovação do ambiente, a distribuição CPU foi reinstalada para os testes finais. A retomada do SBX não utiliza VAE. As versões NumPy/pandas/scikit-learn/PyGAD relevantes à comparação SBX permaneceram iguais. Não se exige que o usuário troque seu Python.

A tentativa inicial de retomada encontrou dois escritores na mesma pasta. A execução antiga foi interrompida; as combinações faltantes foram recalculadas, e a tabela final foi conferida por chave seed/condição/modo, exigindo exatamente 80 linhas únicas. O script agora impede escritores concorrentes, grava o checkpoint de forma atômica e valida a completude antes de emitir o marcador final. A retomada utiliza quatro processos isolados. Uma comparação `quick` com execução sequencial confirmou escores idênticos. Tempos de execução paralela não devem ser comparados como se fossem medições isoladas de desempenho.

```bash
python -m pytest -q
python scripts/review_experiments.py --profile full --phase all
# Alternativamente, executar as fases separadamente:
python scripts/review_experiments.py --profile full --phase sbx --workers 4
python scripts/review_experiments.py --profile full --phase vae
python scripts/review_experiments.py --profile full --phase benchmarks
python scripts/review_experiments.py --profile full --phase controls
# Para continuar a fase SBX interrompida, com mesmo código/configuração e pasta:
python scripts/review_experiments.py --profile full --phase sbx --workers 4 --resume
```

Usar `--output` com uma nova pasta para preservar resultados anteriores. As fases que compartilham a pasta devem ser executadas sequencialmente. Se uma interrupção abrupta deixar `.review.lock`, confirmar que a execução terminou antes de remover esse arquivo. `--resume` verifica perfil, orçamento e configuração, mas não compara automaticamente versões de bibliotecas ou hashes do código; é responsabilidade de quem retoma manter o ambiente equivalente. As variantes VAE estão explicitadas no script. O perfil `quick` reduz gerações e avaliação, mantendo as três seeds exploratórias do VAE.

O notebook principal contém uma execução funcional `quick`, incluindo diagnósticos e exportação, com 11 células de código. O `full` desta revisão foi realizado pelos scripts nos protocolos afetados; não se trata de uma nova execução integral do notebook com todas as seis condições. Os CSVs completos, metadados e figuras estão versionados em `results/review_full_20261019`.
