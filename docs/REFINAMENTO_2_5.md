# Iteração 2.5 — qualidade da geração e concentração seletiva

Protocolo registrado antes da execução full e preservado como referência. A execução foi concluída; consulte [os resultados](RESULTADOS_REFINAMENTO_2_5.md) e [a revisão técnica](REVISAO_2_5.md). A Iris já foi examinada em iterações anteriores: os resultados são exploratórios e não substituem validação numa base independente.

## Diagnóstico de partida

As 25 execuções externas de referência do VAE na 2.4 mostram, em média, diversidade relativa ao treino de 0,723 → 0,123 nos pais selecionados e 0,609 → 0,057 nos candidatos brutos, entre gerações 1 e 40. A diversidade dos sobreviventes muda de 0,929 para 0,715. Essas medidas usam a distância média entre pares, não cobertura percentual de modos. Os valores diferem dos indicadores contra holdout da tabela externa porque a referência de comparação é diferente.

Hipótese: a concentração dos pais e a resposta do decoder contribuem para candidatos pouco diversos. O histórico por si só não demonstra causalidade. O controle de pais fixos congela exatamente os pais selecionados da primeira geração (não todos os dados reais), mantendo tamanho, épocas, batches e número de atualizações do otimizador. Isso isola a mudança dos dados entre gerações; não garante que os pais iniciais sejam plenamente diversos.

## Catálogo congelado

| Família | Referência | Contrastes isolados |
|---|---|---|
| SBX | parâmetros 2.4 | seleção por rank com elitismo 10%; novidade por faixa interquartil; transferência zero; eta 20 |
| VAE-EDA | latent 32, hidden 64, Adam 0,001 | pais iniciais fixos; latent 2; hidden 32; Adam 0,0003 |

Não combinar opções nesta rodada. Beta permanece igual à referência. Cinco nichos fixos e regra probabilística do híbrido permanecem no projeto. Fila histórica do VAE adiada até esclarecer o contraste de pais fixos; não foi implementada nem contabilizada como método concluído.

Na seleção por rank, as cotas de nichos são idênticas às históricas. Dentro de cada cota, floor(0,10 × cota) indivíduos de melhor fitness são preservados; o restante é amostrado sem reposição com pesos de rank lineares. Cotas pequenas podem não ter elite determinística. Nichos sem candidatos continuam usando preenchimento com candidatos restantes. A opção se aplica à seleção moderada por nichos, não ao modo global sem moderação.

A novidade por faixa vale 1 entre os quantis 25% e 75% das distâncias entre reais de treino e seus vizinhos mais próximos. Fora da faixa, decai suavemente com a mesma escala histórica. Não utiliza holdout nem recompensa distâncias ilimitadas.

## Orçamento full

- 5 folds externos, 2 folds internos, mesma divisão da 2.4.
- Triagem: 2 famílias × 5 folds × 5 configurações × 2 folds internos × 3 seeds = 300 jobs; 10 gerações.
- Confirmação: referência + um finalista × 2 famílias × 5 folds × 2 folds internos × 5 seeds = 200 jobs; 40 gerações.
- Teste externo: referência + finalista × 2 famílias × 5 folds × 5 seeds = 100 jobs. Finalista rejeitado é medido, mas não promovido no resultado do procedimento selecionado.
- Controles: inicialização, Gaussiana direta por espécie, mistura evolutiva sem transferência, mistura histórica e produto histórico × 5 folds × 5 seeds = 125 jobs.
- Total: **725 jobs principais**. Perfil quick: 74 jobs exclusivamente funcionais, sem evidência de ganho.

VAE usa 30 épocas por ajuste no full. Seeds: triagem 0–2; confirmação 3–7; externo 20–24. População: 50 por espécie. Manifesto fixa código, ambiente, dados, opções, critérios e orçamento. Retomada com código/ambiente diferente é rejeitada.

## Seleção e avaliação

Mantidos o escore 2.4 (50% candidatos, 50% sobreviventes; sete componentes normalizados apenas com treino), ganho interno mínimo de 5%, guardas por componente e espécie e ausência de aumento de duplicatas exatas. Não promover defaults automaticamente. Utilidade e discriminadores não selecionam hiperparâmetros.

Avaliações secundárias: Random Forest treinada na população sintética completa e testada no holdout real, comparada com treino nos reais e teste no mesmo holdout; proximidade holdout real → treino; métricas por espécie; custo; passos de Adam; KL, unidades ativas e variância do decoder. A comparação de utilidade não iguala o número de registros de treino: mede o uso da população gerada completa e informa explicitamente essa diferença.

A Gaussiana direta é ajustada nos reais de treino por espécie e amostrada sem evolução, clipping ou seleção. Seu escore é somente de amostra final e não deve ser apresentado como escore conjunto de candidato/sobrevivente; comparar sua qualidade com o estágio correspondente dos métodos evolutivos.

Checkpoints 5, 10, 20 e 40 usam os diagnósticos já registrados na mesma trajetória, sem reexecutar treinamento. Nesta rodada são descritivos: duração final fixa em 40. Não escolher parada olhando teste externo.

## Execução

```bash
python -m pytest -q
python scripts/refine_25.py --profile quick --workers 4 --output results/refinement_2_5_quick
python scripts/refine_25.py --profile full --workers 4 --output results/refinement_2_5_full
# Retomada exige o mesmo código e ambiente:
python scripts/refine_25.py --profile full --workers 4 --output results/refinement_2_5_full --resume
```

O notebook deve ler resultados completos, sem disparar silenciosamente a busca full. Arquivos antigos da 2.4 não são sobrescritos.

Após concluir a execução, consolide e audite:

```bash
python scripts/report_25.py results/refinement_2_5_full results/refinement_2_5_summary
python tests/audit_refinement25.py results/refinement_2_5_full
```

A segunda auditoria reconstrói as divisões externas e internas, confere as seeds e o pareamento da confirmação e recalcula as decisões a partir dos resultados internos. O relatório compacto inclui `selection_checks.csv`: cada `False` indica uma guarda não satisfeita, sem alterar seus limites.
