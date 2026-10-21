# Resultados da iteração 2.5

A iteração foi implementada e a avaliação full concluiu **725 execuções**. As novas variantes não produziram ganho adicional no procedimento selecionado em relação à 2.4. A contribuição desta rodada é o diagnóstico das limitações, os controles e a avaliação secundária. Nenhum default foi promovido.

## Comparação com a iteração anterior

Perda conjunta: menor é melhor; média de 25 pares fold–seed por família.

| Procedimento selecionado | 2.4 | 2.5 | Interpretação |
|---|---:|---:|---|
| SBX | 1,453245 | 1,453245 | Referência mantida nos cinco folds; sem ganho adicional |
| VAE-EDA | 2,268258 | 2,269868 | Mesmas escolhas; variação de +0,001610 (~0,07%), sem melhora |

No VAE, Adam 0,0003 foi novamente aceito apenas nos folds externos 2 e 3 (índices iniciados em zero); referência 0,001 nos demais. O ganho de aproximadamente 4% contra o VAE de referência nesta rodada já correspondia à escolha obtida na 2.4: não é ganho novo da 2.5. O SBX continua com a menor perda observada entre os dois procedimentos pesquisados.

A retomada preservou as versões, os hashes de código e o desenho experimental. Três execuções VAE de referência no fold 4 apresentaram pequenas diferenças numéricas em relação ao histórico 2.4, observáveis desde a primeira época (diferença da ordem de 10⁻⁷ na perda inicial), amplificadas durante a evolução. A causa específica não foi isolada; não atribuímos essa variação a uma mudança de hiperparâmetro nem prometemos equivalência bit a bit entre sessões. A comparação pareada completa está em `comparison_2_4.csv`.

## O que foi aprendido

### 1. A concentração do VAE não depende apenas da perda de diversidade dos pais

Na triagem interna, com 10 gerações e **900 atualizações de Adam** por execução, os contrastes isolados produziram:

| Variante VAE | Diversidade relativa dos pais | Diversidade relativa dos candidatos brutos |
|---|---:|---:|
| Referência | 0,2140 | 0,1599 |
| Pais iniciais fixos | 0,6628 | 0,2142 |
| Hidden 32 | 0,2298 | 0,1532 |
| Latent 2 | 0,1984 | 0,0479 |
| Adam 0,0003 | 0,3924 | 0,3366 |

Médias sobre os folds internos, seeds e espécies. A referência dessas razões são os reais de treino, diferentemente das métricas externas contra holdout. Não são percentuais de modos cobertos.

Congelar os pais preserva a diversidade da entrada, mas os candidatos continuam concentrados. Isso é consistente com contribuição tanto da seleção parental quanto do comportamento do modelo; não identifica sozinho a causa neural. Nos históricos 2.4, KL, unidades ativas e variância de saída do decoder também caíam durante a evolução. Reduzir a dimensão latente para 2 piorou a concentração neste contraste; não houve benefício que justificasse promover uma arquitetura menor.

Pais fixos chegaram à confirmação nos folds 0 e 4, mas foram rejeitados por guardas de diversidade dos sobreviventes e quase-cópias dos candidatos em determinadas espécies. Os detalhes estão em `failed_species_guards.csv`. Uma fila histórica do VAE permanece adiada: preservar dados anteriores, isoladamente, não demonstrou resolver o problema com as guardas vigentes.

### 2. Melhor média não garante melhoria aceitável por espécie

No SBX, eta 20 foi finalista nos folds 0–3 e novidade por faixa no fold 4. Nenhum passou por todas as exigências da confirmação. Seleção por rank e transferência zero foram avaliadas na triagem e não avançaram. Isso vale para as configurações e o orçamento testados, não demonstra inutilidade geral desses mecanismos.

As guardas, o ganho mínimo interno de 5% e os pesos do escore permaneceram iguais à 2.4. Cada rejeição é registrada em `selection_checks.csv`, complementada pelos excessos por espécie em `failed_species_guards.csv`.

### 3. A transferência gaussiana explica um compromisso importante

Controles externos com 25 execuções por condição:

| Mistura evolutiva | Perda conjunta | Incoerência dos candidatos | Quase-cópias dos candidatos | Diversidade relativa dos candidatos |
|---|---:|---:|---:|---:|
| Mistura histórica | 2,2932 | 36,0% | 13,6% | 1,4152 |
| Apenas a própria espécie | 1,8031 | 0,0% | 19,1% | 0,3088 |

Retirar a transferência reduz a perda conjunta em cerca de 21,4% neste controle, mas aumenta a taxa de quase-cópias e concentra fortemente os candidatos. Essa condição foi um comparador externo predefinido, sem seleção interna para promoção; não virou default. Uma boa média pode ocultar compromissos relevantes.

### 4. O controle sem evolução merece permanecer na avaliação

A Gaussiana direta por espécie, ajustada apenas nos reais de treino, apresentou perda de **amostra final 1,3286**, diversidade relativa 1,0552, quase-cópias 11,7% e incoerência 3,1%. Como não possui ciclo evolutivo, sua perda não é o composto candidato/sobrevivente: compare-a com o estágio de amostra final dos outros métodos (SBX 1,1339; VAE selecionado 1,5859).

A Random Forest real versus sintético teve acurácia de 55,1% nesse controle. Aproximação de 50% não comprova privacidade, criatividade ou superioridade global. Nos reais reservados, a taxa de proximidade ao treino pelo mesmo limiar foi 5,3%; isso contextualiza, mas não elimina, as quase-cópias sintéticas.

## Utilidade dos sintéticos

Acurácia balanceada para prever a espécie em reais reservados, com uma Random Forest fixa:

| Dados usados para treinar o classificador | Acurácia balanceada |
|---|---:|
| Reais de treino | 95,3% |
| Sintéticos SBX selecionado | 94,5% |
| Sintéticos VAE selecionado | 95,3% |
| Gaussiana direta | 96,8% |

Essas diferenças são descritivas, sem declaração de superioridade estatística. Os sintéticos usam a população completa de 150 indivíduos; o treino real externo contém 120. O teste real é o mesmo. Esta métrica é secundária e não orientou a busca. Não confundir esse classificador de espécies com o discriminador real/sintético.

## Entrega e validação

- Na entrega inicial, 26 testes passaram; comparação funcional dos defaults reproduziu exatamente as populações do commit anterior em quatro famílias no cenário de regressão documentado.
- Após a revisão técnica, 28 testes passaram, incluindo elitismo total e auditoria sob `python -O`; a validação local também aprovou os 28 testes. As auditorias foram reaplicadas e 22 arquivos consolidados reproduzidos byte a byte, sem repetir a geração. Ver [revisão técnica](REVISAO_2_5.md).
- Perfil quick final: 74 jobs, auditoria de integridade e de divisões/decisões aprovada; não usado para inferir desempenho.
- Full: 300 triagens, 200 confirmações, 100 avaliações externas e 125 controles.
- Auditorias verificaram hashes dos jobs e do código, perdas finitas, índices disjuntos, reconstrução dos folds/seeds, pareamento da confirmação e decisões recalculadas apenas com dados internos.
- Custo somado dos jobs: cerca de 72 minutos de tempo decorrido por job; não é tempo de parede total nem uma medida de CPU, porque houve paralelismo e retomada.
- Notebook atualizado: seção 8 lê o resumo salvo. Apenas essa nova célula de leitura foi executada para registrar os resultados; a suíte histórica do notebook não foi reexecutada nesta entrega.

Fontes de evidência: `results/refinement_2_5_summary/`, `results/refinement_2_5_diagnosis/` e arquivo dos jobs completos. Protocolo: `docs/REFINAMENTO_2_5.md`.

A Iris já foi examinada em várias iterações. Esta rodada mantém avaliação aninhada, mas não constitui validação inédita numa população independente. As evidências ajudam a decidir a próxima hipótese e não justificam retirar guardas ou ampliar indefinidamente a busca para obter um resultado favorável.
