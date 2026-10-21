# Resultados do refinamento 2.4

## Estado e conclusão

A avaliação aninhada completa foi concluída e auditada: **3265 execuções**, seis famílias, cinco folds externos, cinco seeds externas e 150 comparações pareadas. Em **3 das 30 decisões** família/fold, a seleção interna aceitou uma alternativa à referência. Os padrões de geração permanecem iguais aos da versão 2.3.1.

O sinal mais consistente neste protocolo foi o VAE: a perda externa média caiu aproximadamente **4,0%**, com taxa de aprendizado 0,0003 escolhida em dois folds. O híbrido melhorou aproximadamente **1,4%**, mas seu intervalo condicional inclui zero e houve aumento médio do erro de diversidade dos sobreviventes (Δ=+0,012). Isso não sustenta uma promoção geral das alternativas. A recomendação é manter os padrões e preservar essas variantes como opções experimentais para uma avaliação futura em dados independentes.

A avaliação mede o procedimento de seleção completo: cada fold escolhe sua configuração usando somente dados internos. Quando a referência é escolhida, a diferença externa é zero por construção; isso não demonstra equivalência entre todas as alternativas rejeitadas.

## Referência versus procedimento selecionado

Perda menor é melhor. O objetivo combina 50% da perda dos candidatos e 50% da perda dos sobreviventes. Cada média usa os mesmos 25 pares fold/seed por família. A coluna Δ é selecionado menos referência.

| Família | Perda referência | Perda selecionada | Δ perda | Intervalo condicional 95% | Folds com mudança |
| --- | --- | --- | --- | --- | --- |
| SBX | 1,453 | 1,453 | 0,000 | [0,000; 0,000] | 0/5 |
| OB-Scan | 2,043 | 2,043 | 0,000 | [0,000; 0,000] | 0/5 |
| Híbrido probabilístico | 1,534 | 1,513 | -0,021 | [-0,047; 0,005] | 1/5 |
| Mistura gaussiana | 2,293 | 2,293 | 0,000 | [0,000; 0,000] | 0/5 |
| Produto gaussiano | 2,905 | 2,905 | 0,000 | [0,000; 0,000] | 0/5 |
| VAE-EDA | 2,363 | 2,268 | -0,095 | [-0,117; -0,069] | 2/5 |

O intervalo bootstrap reamostra as cinco seeds após agregar os folds. Ele descreve somente aleatoriedade algorítmica condicionada à Iris e às divisões fixas; não é intervalo populacional nem demonstra generalização. Um intervalo [0; 0] decorre da escolha da própria referência em todos os folds.

## Controle sem evolução

A inicialização não gera candidatos. Por isso, nesta tabela são comparadas somente perdas de sobreviventes. RF é um diagnóstico de distinguibilidade real/sintético, não o critério de seleção; aproximação de 50% isoladamente não comprova fidelidade.

| Família | Perda inicialização | Perda referência | Perda selecionada | RF inicialização | RF referência | RF selecionada |
| --- | --- | --- | --- | --- | --- | --- |
| SBX | 2,447 | 1,134 | 1,134 | 56,1% | 74,5% | 74,5% |
| OB-Scan | 2,447 | 1,392 | 1,392 | 56,1% | 73,1% | 73,1% |
| Híbrido probabilístico | 2,447 | 1,177 | 1,167 | 56,1% | 77,6% | 77,3% |
| Mistura gaussiana | 2,447 | 1,227 | 1,227 | 56,1% | 69,1% | 69,1% |
| Produto gaussiano | 2,447 | 1,239 | 1,239 | 56,1% | 63,9% | 63,9% |
| VAE-EDA | 2,447 | 1,626 | 1,585 | 56,1% | 71,9% | 71,1% |

Os controles de inicialização são compartilhados conceitualmente entre famílias; suas repetições não são seis conjuntos independentes de evidência. A RF usa CV discriminativa própria de três folds dentro da avaliação externa, separada dos folds que selecionam o gerador.

## Candidatos, sobreviventes e proximidade ao treino

| Família | Perda candidatos | Perda sobreviventes | Proximidade candidatos | Proximidade sobreviventes | Proximidade inicialização | Duplicação exata sobreviventes |
| --- | --- | --- | --- | --- | --- | --- |
| SBX | 1,773 | 1,134 | 3,6% | 3,3% | 55,2% | 0,0% |
| OB-Scan | 2,694 | 1,392 | 1,9% | 12,0% | 55,2% | 0,0% |
| Híbrido probabilístico | 1,858 | 1,167 | 3,1% | 5,6% | 55,2% | 0,0% |
| Mistura gaussiana | 3,359 | 1,227 | 13,6% | 5,5% | 55,2% | 0,0% |
| Produto gaussiano | 4,570 | 1,239 | 4,9% | 7,2% | 55,2% | 0,0% |
| VAE-EDA | 2,952 | 1,585 | 4,3% | 19,5% | 55,2% | 0,0% |

Proximidade usa o percentil 5 das distâncias entre vizinhos reais no treino de cada execução. É um diagnóstico descritivo, não uma garantia de privacidade. Os candidatos são a última geração proposta; para crossovers, o operador já aplica os limites e não há amostra anterior a essa aplicação. Para EDAs, os diagnósticos registram também o comportamento das propostas antes das correções de limites; o arquivo não contém todos os vetores candidatos de todas as gerações.

## Configurações selecionadas internamente

| Família | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | --- | --- | --- | --- | --- |
| SBX | reference | reference | reference | reference | reference |
| OB-Scan | reference | reference | reference | reference | reference |
| Híbrido probabilístico | reference | tournament2 | reference | reference | reference |
| Mistura gaussiana | reference | reference | reference | reference | reference |
| Produto gaussiano | reference | reference | reference | reference | reference |
| VAE-EDA | reference | reference | refined_learning_rate | refined_learning_rate | reference |

`reference` mantém a configuração anterior; `tournament2` reduz o torneio de três para dois participantes; `refined_learning_rate` usa taxa de aprendizado 0,0003 no VAE, mantendo o restante das opções da base elegível. As opções completas de todas as decisões estão nos arquivos `selection_*.json`. As guardas exigem ganho interno mínimo de 5% e limitam deterioração por métrica, espécie e estágio, além de impedir aumento de duplicação exata. Não foram flexibilizadas depois de observar os resultados externos.

## O que mudou e o que não se pode concluir

- Foram implementadas opções explícitas de fitness, torneio, pais por nichos, produtos condicionados à espécie, warmup de KL e reamostragem limitada, além de diagnósticos e busca aninhada retomável.
- A avaliação separa qualidade de geração e sobrevivência e compara evolução com inicialização. O ganho de instrumentação e controle experimental não implica, por si só, ganho de qualidade dos sintéticos.
- Os números históricos enviados ao Saulo usavam outro orçamento e outro protocolo. Não representam uma comparação direta antes/depois com esta tabela. Nenhuma redução de acurácia histórica é atribuída automaticamente às alterações novas.
- A Iris já foi explorada anteriormente; o estudo continua exploratório. População de 50 por espécie, 40 gerações, cinco seeds externas e somente 150 registros reais limitam as conclusões.
- Nichos dinâmicos e divisão exata de 50/50 continuam fora do escopo. O híbrido segue probabilístico.

## Validação e reprodução

**22 testes passaram** no ambiente restaurado. A célula final do notebook foi executada sem erros, com quatro saídas de resultados preservadas; as células históricas não foram reexecutadas nesta consolidação. A auditoria independente verificou hashes dos jobs, finitude, populações, gerações, pareamento, cobertura dos folds e ausência de índices externos nas subdivisões internas. Também foi conferida a identidade dos arquivos de implementação com os hashes do manifesto. Essas verificações não provam superioridade científica.

Implementação executada: commit remoto `ab11a36863b02c60f545a4944f2315a5a67ba3e9`, árvore `07336f732da4545c53a93d4f88a00b657c659906`. Python 3.12.14; versões completas em `manifest.json`. A execução foi retomada após interrupção com identidade de código, dados, configurações e ambiente preservada. Tempo em paralelo não deve ser interpretado como benchmark isolado de velocidade.

As tabelas compactas, decisões, manifesto, conclusão e auditoria estão em `results/refinement_2_4_summary/`. Os jobs completos são preservados separadamente em um arquivo de evidências; não foram adicionados milhares de arquivos ao Git. O notebook lê a pasta completa quando disponível e a consolidação versionada como alternativa.

Para nova execução independente, use o comando do [protocolo](REFINAMENTO_2_4.md) com uma pasta nova. Para auditar o arquivo completo extraído: `python tests/audit_refinement_results.py results/refinement_nested_full_20261020`.
