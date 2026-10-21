# Iris Generativa com NatGenAI

Experimento de geração de dados sintéticos inspirado em [Evolutionary Computation as Natural Generative AI](https://arxiv.org/html/2510.08590v1). A iteração **2.5 está concluída e integrada à `main`** (versão do pacote: `2.5.0`). A suíte principal compara seis condições, mantendo 50 registros por espécie da Iris: **SBX, OB-Scan-KDE, híbrido probabilístico, mistura gaussiana, mistura com produtos gaussianos e VAE-EDA condicional**.

O notebook principal é [`notebooks/Iris_Generativa_NatGenAI.ipynb`](notebooks/Iris_Generativa_NatGenAI.ipynb). Ele contém a explicação dos métodos, a pipeline, saídas históricas da suíte e as consolidações `full` dos refinamentos 2.4 e 2.5 nas seções 7 e 8. Essas duas seções leem resultados salvos, sem repetir a busca. As diferenças entre artigo e implementação estão em [docs/METODOLOGIA.md](docs/METODOLOGIA.md).

## Executar

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
jupyter notebook notebooks/Iris_Generativa_NatGenAI.ipynb
```

Para instalar e testar no ambiente Windows/Python 3.14 existente:

```powershell
python -m pip install --only-binary=:all: -r requirements-validation.txt
python -m pip check
python -m pytest -q
```

`requirements-validation.txt` reutiliza `requirements.txt` e acrescenta pytest; não força o pandas 2.2.3 do ambiente anterior. A opção `--only-binary=:all:` evita compilação local e informa se algum pacote não possui distribuição compatível. Não é necessário trocar de versão do Python para tentar essa instalação. PyTorch é usado em CPU; não há exigência de GPU.

O ambiente Linux/Python 3.12 efetivamente usado nos testes anteriores foi preservado como registro histórico em `docs/environment-validation-py312.txt`; não use esse registro para instalar no Windows/Python 3.14. A execução nessa plataforma deve ser confirmada pelos testes locais.

Em `iris_generativa/config.py`, cada hiperparâmetro tem comentário explicativo. Configure antes de importar o pacote e reinicie o kernel após mudanças:

| Perfil | Gerações Iris | Épocas VAE por ajuste | Seeds de robustez | CV discriminativa | Árvores RF |
|---|---:|---:|---:|---|---:|
| `quick` (padrão) | 3 | 3 | 2 | 2 folds × 1 repetição | 32 |
| `full` | 40 | 30 | 20 | 5 folds × 5 repetições | 300 |

No protocolo histórico da suíte, o cross-fitting mantém cinco folds externos e três internos nos dois perfis. A população principal permanece em 150 indivíduos; no cross-fitting, são 30 sintéticos por fold. `RUN_*` permite desligar laboratórios individualmente. `SHOW_PROGRESS=False` oculta barras sem alterar os resultados.

**O perfil rápido verifica funcionamento e não estabelece superioridade entre métodos.** A execução `full` enviada pelo usuário está preservada em `results/baseline_full_20261019`. A revisão 2.3.1 corrige a assimetria do SBX e reavalia os protocolos afetados em orçamento completo; resultados e limites estão em [docs/REVISAO_METODOLOGICA.md](docs/REVISAO_METODOLOGICA.md).

## Estado atual e protocolos

A validação da 2.5 concluiu **28 testes**, **725 execuções full** e auditorias de integridade, divisões e decisões. A validação local também passou: 28 testes e leitura da seção 8. O [PR #3](https://github.com/fmfelipemartins/iris-generativa-natgenai/pull/3) foi integrado em 21/10/2026, no commit `e06406a`. Os defaults de geração continuam os da referência 2.3.1.

O cross-fitting histórico da suíte e a avaliação aninhada dos refinamentos são protocolos distintos. No refinamento 2.5 full, são cinco folds externos, dois internos e população evolutiva constante de 150 indivíduos. A triagem e a confirmação escolhem variantes usando apenas subdivisões internas; o teste externo é avaliado depois da decisão. Os controles externos e a avaliação de utilidade não escolhem hiperparâmetros.

## Pipeline

![Pipeline da suíte principal](docs/pipeline.svg)

1. Carregar dados, padronizar, calibrar fitness e formar cinco nichos por tarefa.
2. Gerar candidatos por crossover ou ajustar um modelo aos pais selecionados e amostrar.
3. Selecionar sobreviventes por tarefa, opcionalmente preservando nichos; repetir gerações.
4. Medir duplicação, coerência, novidade, diversidade, fidelidade marginal e distinguibilidade.
5. Comparar seeds, sensibilidade, políticas de seleção e cenários artificiais 2D.
6. Repetir o ajuste dentro de folds com reais reservados para avaliação.

Os refinamentos acrescentam seleção parental por nichos, sobrevivência probabilística por rank e novidade por faixa como alternativas experimentais. A avaliação aninhada separa triagem, confirmação e teste externo; registra candidatos e sobreviventes, critérios de rejeição por espécie, controles gaussianos e utilidade em classificação de espécies. O diagrama acima resume a suíte principal; a [pipeline da 2.4](docs/pipeline_refinamento.svg) registra aquela busca, e o [fluxo da 2.5](docs/METODOLOGIA.md#fluxo-da-avaliação-25) descreve o protocolo final. As diferenças entre fitness, perda conjunta, indistinguibilidade e utilidade estão em [Metodologia](docs/METODOLOGIA.md).

O fitness permanece `0,50 P + 0,25 N + 0,25 C`. Na configuração padrão, a novidade favorece uma distância-alvo ao real mais próximo; não cresce indefinidamente com a distância. O híbrido sorteia o operador por descendente, sem cota exata de 50/50. Nichos dinâmicos permanecem fora desta iteração.

## Implementação

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Hiperparâmetros, perfis e flags comentados |
| `data.py`, `fitness.py` | Dados, calibração e fitness por tarefa |
| `operators.py`, `evolution.py` | SBX simétrico, modo legado de auditoria, OB-Scan e sobrevivência |
| `gaussian.py` | Gaussianas por tarefa, misturas e produtos normalizados |
| `vae.py` | VAE condicional, treinamento e amostragem com RNG privado |
| `generative.py` | Interface comum e ciclo EDA |
| `benchmarks.py` | Cenários 2D, figuras e controle contrafactual da seleção |
| `evaluation.py` | Métricas e discriminadores |
| `extended_experiments.py` | Seis condições, robustez, sensibilidade e cross-fitting |
| `experiments.py` | API anterior preservada, agora com barras de progresso |
| `progress.py` | Exibição de progresso com tqdm |
| `diagnostics.py`, `artifacts.py` | Diagnósticos por estágio e exportação de execuções |
| `scripts/review_experiments.py` | Auditoria SBX e comparações controladas da revisão |
| `scripts/refine_experiments.py`, `scripts/refine_25.py` | Avaliações retomáveis dos refinamentos 2.4 e 2.5 |
| `scripts/report_25.py` | Auditoria de integridade e consolidação da 2.5 |
| `tests/audit_refinement_results.py`, `tests/audit_refinement25.py` | Auditorias das avaliações aninhadas |
| `tests/test_refinement.py`, `tests/test_refinement25.py` | Testes dos refinamentos e casos de regressão |
| `tests/test_iteration.py` | Regressão, referências analíticas e isolamento |

As saídas incluem histórico de fitness, avaliações de fitness e tempo de geração. Novos geradores também registram componente de origem, clipping e sobrevivência dos candidatos; o VAE registra perdas por época. Os resultados da suíte guardam perfil, configuração, seeds e versões. A célula de registro da execução exporta tabelas CSV, amostras e diagnósticos em uma pasta exclusiva `results/execution_<perfil>_<data>`. Essas exportações locais são ignoradas pelo Git; referências selecionadas são versionadas separadamente.

O SBX padrão agora sorteia os dois ramos por gene. `SBX_MODE="legacy_lower"` existe apenas para reproduzir o comportamento assimétrico anterior. A instrumentação do VAE separa candidatos e sobreviventes, acompanha fundadores, KL, unidades ativas e resposta do decoder. Na revisão histórica 2.3.1, o cross-fitting não escolheu hiperparâmetros de treinamento. Nos refinamentos 2.4 e 2.5, a escolha ocorre nas subdivisões internas da avaliação aninhada, sem consultar os resultados externos para decidir.

## Verificar

```bash
python -m pip install pytest
python -m pytest -q
```

O teste de regressão compara as três condições anteriores, com SBX em modo `legacy_lower`, com o commit `ff2659f`; exige um clone com esse histórico Git. Os demais testes verificam os produtos gaussianos, reprodutibilidade, aprendizado do VAE, cotas, controles de seleção e exclusão dos dados reservados no cross-fitting.

Consulte [a revisão técnica 2.5](docs/REVISAO_2_5.md) para a validação atual e [VALIDACAO.md](VALIDACAO.md) para o histórico. O uso da Iris é experimental: fidelidade e baixa distinguibilidade não demonstram validade biológica, privacidade ou criatividade fora da distribuição.

## Refinamento 2.4

A iteração 2.4 acrescentou seleção parental por nichos, produtos condicionados à espécie, warmup de KL, diagnósticos comuns e avaliação aninhada com população evolutiva constante. Os padrões de geração da versão 2.3.1 permanecem como referência; nenhuma variante é promovida automaticamente.

Consulte [o protocolo e seus limites](docs/REFINAMENTO_2_4.md). Para executar a avaliação aninhada: `python scripts/refine_experiments.py --profile full --phase nested --output results/refinement_local_full --workers 4`. Essa busca é mais custosa que executar uma vez o notebook. Resultados parciais são salvos por job.

A avaliação completa desta iteração e sua interpretação estão em [Resultados do refinamento 2.4](docs/RESULTADOS_REFINAMENTO_2_4.md). A seção 7 do notebook também lê a consolidação versionada em `results/refinement_2_4_summary`, sem executar a busca novamente. Os registros individuais completos são preservados no arquivo de evidências da execução; a consolidação contém as tabelas externas, as decisões de seleção e a auditoria.

## Refinamento 2.5

A iteração 2.5 isola os efeitos da seleção por rank, da novidade por faixa e da concentração dos pais do VAE. O protocolo limita a busca principal a 725 execuções e mantém os padrões históricos. Inclui uma Gaussiana direta sem evolução, controles de transferência, utilidade dos sintéticos e proximidade dos reais reservados.

Consulte [o protocolo 2.5](docs/REFINAMENTO_2_5.md). Execute com `python scripts/refine_25.py --profile full --workers 4 --output results/refinement_2_5_local_full`. Ao concluir, use `python scripts/report_25.py results/refinement_2_5_local_full results/refinement_2_5_local_summary` e `python tests/audit_refinement25.py results/refinement_2_5_local_full`. A seção 8 do notebook lê o resumo sem iniciar a busca.

A avaliação está consolidada em [Resultados do refinamento 2.5](docs/RESULTADOS_REFINAMENTO_2_5.md). Não houve ganho adicional no procedimento selecionado em relação à 2.4; os contrastes esclareceram a concentração do VAE e os compromissos da transferência gaussiana.

Use uma pasta nova para cada execução com código ou configuração diferentes. Os manifestos históricos correspondem ao commit `bb41d38661b159f7f1ae094ad73ed13c470db45f`, anterior às correções da revisão técnica; não altere seus hashes para retomar com código diferente. Consulte [a proveniência](docs/REVISAO_2_5.md). Para consultar os resultados existentes, basta abrir a seção 8 do notebook; não é necessário executar novamente o perfil full.
