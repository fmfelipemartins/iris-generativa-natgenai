# Validação da iteração 2.3 — 19/10/2026

A implementação foi validada no perfil **quick**. O protocolo científico completo (`full`) não foi executado nesta entrega; os resultados abaixo são verificações funcionais, não evidência de superioridade dos métodos.

## Verificações executadas

- `python -m pytest -q`: **12 testes aprovados**. Cobertura: produto gaussiano analítico e amostragem; covariâncias degeneradas; componentes por pares/triplo; regressão exata dos três operadores anteriores contra `ff2659f`; reprodutibilidade, cotas e limites dos três novos geradores; aprendizado do VAE e preservação do RNG global Torch; invariância ao progresso; controle global/por tarefa/nichos; cenários 2D; ausência dos reais reservados no ajuste do cross-fitting.
- `python -m ruff check iris_generativa tests --select F`: aprovado.
- `git diff --check`: aprovado.
- **Nove células de código do notebook executadas sequencialmente, sem erros**, em um único processo IPython, com captura das tabelas e figuras. O lançamento de kernel TCP pelo nbclient foi bloqueado pelo ambiente; a execução sequencial no processo permitiu verificar todas as células e salvar as saídas. Não foi validada uma sessão interativa de Jupyter neste ambiente.
- Notebook: seis condições principais, 12 execuções de robustez (duas seeds × seis condições), cinco casos de sensibilidade, duas políticas de seleção Iris, laboratório polinomial, 12 combinações cenário/método 2D, nove pontos de trajetória e 27 decisões contrafactuais, 30 execuções de cross-fitting (cinco folds × seis condições).
- Todas as condições principais produziram matrizes finitas de 150 × 4, com 50 registros por espécie e zero duplicações exatas dos reais. As figuras e a pipeline foram inspecionadas.

## Ambiente utilizado

Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; Matplotlib 3.10.8; scikit-learn 1.8.0; SciPy 1.17.0; PyGAD 3.7.0; PyTorch 2.14.1; tqdm 4.70.1. O VAE utilizou CPU. As versões desse ambiente estão registradas em `docs/environment-validation-py312.txt`. A pilha original de dependências foi mantida em `requirements.txt` e não foi integralmente reinstalada nesta validação.

## Observações funcionais do perfil rápido

No cenário 2D de seleção, a política global terminou com 62/18 indivíduos por tarefa, enquanto as políticas por tarefa e por tarefa com nichos mantiveram 40/40. Isso demonstra que o controle implementado permite perda de representação e que as cotas a evitam neste cenário.

O VAE reduziu a perda de reconstrução ao longo das nove épocas totais da execução Iris rápida, mas somente sete de seus 150 candidatos da última geração sobreviveram. A população final ainda contém muitos indivíduos provenientes da inicialização ou de gerações anteriores. Portanto, uma baixa distinguibilidade dessa população não pode ser atribuída isoladamente à qualidade do VAE. No cenário 2D, o orçamento reduzido também produz amostras concentradas; convergência e diversidade exigem o protocolo completo e análise própria.

Os escores e tempos desta execução estão nas saídas do notebook. Tempos dependem do ambiente e as comparações entre métodos não igualam orçamento de treinamento. A política global, a geometria anel/espiral e a adaptação para três espécies são controles e adaptações documentados em `docs/METODOLOGIA.md`.

---

# Histórico da validação anterior (2.2)

O conteúdo abaixo pertence à versão anterior e não representa resultados dos três novos geradores nem da execução rápida desta iteração.

# Validação do projeto Iris Generativa com NatGenAI

## 1. Objetivo da validação

Este documento registra a validação técnica e operacional do projeto **Iris Generativa com NatGenAI** após a reorganização da implementação em um pacote Python modular.

A validação busca confirmar que:

- a estrutura do projeto está coerente para uso local e versionamento no GitHub;
- o pacote `iris_generativa` pode ser instalado e importado corretamente;
- o notebook principal executa utilizando o ambiente virtual configurado;
- as dependências utilizadas são compatíveis com o projeto;
- a separação entre notebook e módulos Python não introduziu erros estruturais;
- os arquivos gerados automaticamente pelo ambiente não fazem parte do código-fonte versionado.

---

## 2. Estrutura validada

A estrutura principal do projeto é:

```text
iris-generativa-natgenai/
├── .gitignore
├── iris_generativa/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── evolution.py
│   ├── experiments.py
│   ├── fitness.py
│   └── operators.py
│
├── notebooks/
│   └── Iris_Generativa_NatGenAI.ipynb
│
├── pyproject.toml
├── README.md
├── requirements.txt
└── VALIDACAO.md
```

A pasta `.venv/` é utilizada localmente, mas não faz parte do repositório.

Também foram removidos do projeto os artefatos gerados automaticamente durante a execução e instalação, como:

```text
__pycache__/
*.egg-info/
```

Esses itens estão cobertos pelo `.gitignore`.

---

## 3. Ambiente de execução validado

O ambiente virtual foi criado localmente com:

```text
Python: 3.14.7
NumPy: 2.5.3
pandas: 3.0.6
scikit-learn: 1.9.1
PyGAD: 3.7.0
```

As dependências foram instaladas a partir de:

```text
requirements.txt
```

O pacote local foi instalado em modo editável com:

```powershell
python -m pip install -e . --no-deps
```

A importação do pacote foi testada com:

```powershell
python -c "import iris_generativa; print('Pacote instalado corretamente')"
```

Resultado observado:

```text
Pacote instalado corretamente
```

Isso confirma que o pacote `iris_generativa` está acessível pelo ambiente virtual utilizado no notebook.

---

## 4. Kernel e notebook

O notebook principal utiliza o kernel correspondente ao ambiente virtual `.venv`.

Arquivo validado:

```text
notebooks/Iris_Generativa_NatGenAI.ipynb
```

O notebook possui:

- **40 células no total**;
- **24 células Markdown**;
- **16 células Python**.

Antes da validação operacional, o kernel foi reiniciado para eliminar variáveis, objetos e estados residuais de execuções anteriores.

Em seguida, o notebook foi executado novamente a partir do início.

A execução avançou sequencialmente até a última célula Python, correspondente ao protocolo cross-fitted, registrada como execução **[16]**, com resultado produzido.

Isso fornece evidência de que o notebook pode ser executado a partir de um kernel limpo utilizando a estrutura modular atual.

---

## 5. Validação dos módulos Python

Os módulos do pacote estão separados por responsabilidade:

| Módulo | Responsabilidade |
|---|---|
| `config.py` | Hiperparâmetros, seeds e flags de execução |
| `data.py` | Carregamento da Iris, transformação dos dados e contexto experimental |
| `fitness.py` | Plausibilidade, novidade, coerência e fitness generativo |
| `operators.py` | SBX, OB-Scan-KDE e operadores auxiliares |
| `evolution.py` | Inicialização populacional, seleção e processo evolutivo |
| `evaluation.py` | Métricas, fidelidade estatística e discriminadores |
| `experiments.py` | Protocolos experimentais e experimentos complementares |

Na revisão estrutural anterior, os módulos e as células Python do notebook foram verificados quanto à sintaxe, sem identificação de erros de compilação.

A reorganização do projeto separou a implementação de baixo nível da narrativa experimental, mantendo o notebook como documento principal de execução, visualização e interpretação.

---

## 6. Preservação metodológica

A reorganização estrutural foi realizada com o objetivo de melhorar:

- legibilidade;
- modularidade;
- manutenção;
- apresentação ao orientador;
- preparação para versionamento em GitHub.

A refatoração não teve como objetivo alterar:

- função de fitness;
- pesos do fitness;
- hiperparâmetros;
- condições experimentais;
- operadores principais;
- métricas;
- classificadores;
- protocolo de validação;
- protocolo cross-fitted.

As alterações recentes no notebook foram predominantemente editoriais e de organização das células Markdown.

Os módulos Python permaneceram preservados durante essas alterações.

---

## 7. Experimentos disponíveis

O projeto contém:

### Experimento principal

Comparação entre:

- SBX;
- OB-Scan;
- condição híbrida 50/50.

### Robustez

Repetição das condições em múltiplas seeds.

### Sensibilidade do SBX

Comparação entre diferentes valores de:

```text
eta = 5
eta = 30
eta = 50
```

### RMP

Comparação do comportamento com diferentes configurações de transferência intertarefas.

### Seleção moderada

Avaliação do efeito da preservação de diversidade populacional.

### SBX + Polynomial Mutation

Experimento separado para avaliar a introdução de mutação polinomial sem confundir a comparação principal entre operadores.

### Cross-fitting

Avaliação dos dados sintéticos contra registros reais que não foram utilizados como referência durante a geração.

---

## 8. Resultados de referência

Os resultados armazenados e analisados durante o desenvolvimento incluem, entre outros, os seguintes valores médios no experimento de robustez com múltiplas seeds:

### Random Forest — acurácia média

| Condição | Acurácia média |
|---|---:|
| SBX | 0.8610 |
| OB-Scan | 0.8813 |
| Híbrido 50/50 | 0.8753 |

### Cross-fitting — acurácia média

| Condição | Acurácia média |
|---|---:|
| SBX | 0.7300 |
| OB-Scan | 0.7567 |
| Híbrido 50/50 | 0.7400 |

Esses valores são mantidos como referências experimentais para comparação com futuras alterações metodológicas.

Uma refatoração estrutural não deve ser considerada válida se modificar resultados de referência sem uma alteração metodológica explicitamente documentada.

---

## 9. Arquivos ignorados pelo Git

O `.gitignore` foi configurado para impedir o versionamento de arquivos locais ou gerados automaticamente, incluindo:

```text
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
.vscode/
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/
```

Isso mantém o repositório concentrado no código-fonte, documentação, configuração e notebook experimental.

---

## 10. Estado da validação

| Item | Estado |
|---|---|
| Estrutura modular do pacote | ✅ Validada |
| Ambiente virtual | ✅ Validado |
| Dependências | ✅ Instaladas |
| PyGAD 3.7.0 | ✅ Validado |
| Instalação editável do pacote | ✅ Validada |
| Importação de `iris_generativa` | ✅ Validada |
| Kernel `.venv` no notebook | ✅ Validado |
| Reinicialização do kernel | ✅ Realizada |
| Execução sequencial do notebook | ✅ Realizada |
| Última célula experimental | ✅ Executada |
| Separação notebook/pacote | ✅ Validada |
| README | ✅ Atualizado |
| `.gitignore` | ✅ Criado |
| `__pycache__` e `*.egg-info` | ✅ Removidos do projeto |
| Preparação estrutural para GitHub | ✅ Concluída |

---

## 11. Escopo desta validação

Esta validação confirma a integridade estrutural e a execução operacional do projeto no ambiente local utilizado.

Ela não substitui:

- testes unitários formais;
- integração contínua;
- validação estatística inferencial dos resultados;
- revisão científica externa;
- comparação bit a bit de todos os outputs entre versões históricas.

Mudanças futuras que alterem fitness, operadores, hiperparâmetros, protocolos experimentais ou métricas devem ser registradas como **alterações metodológicas**, e não apenas como refatorações estruturais.

---

## 12. Conclusão

O projeto encontra-se organizado como um pacote Python modular, com ambiente reproduzível, notebook principal executável e documentação adequada para versionamento.

A separação entre implementação e notebook permite que:

- o pacote concentre a lógica reutilizável;
- o notebook concentre a narrativa científica;
- os hiperparâmetros permaneçam centralizados;
- novos experimentos possam ser adicionados sem aumentar desnecessariamente a complexidade visual do notebook.

Com a validação operacional concluída e os artefatos locais excluídos do versionamento, o projeto está estruturalmente preparado para ser publicado em um repositório Git.


## Correção de instalação no Windows/Python 3.14

O instalador de validação anterior fixava pandas 2.2.3, que levou o pip no computador do usuário a tentar compilar o pacote e falhar por falta das ferramentas Visual Studio. `requirements-validation.txt` agora referencia as dependências principais e acrescenta pytest. O registro histórico do ambiente Linux/Python 3.12 foi separado em `docs/environment-validation-py312.txt`. Isso corrige a divergência dos arquivos de instalação, sem alterar os algoritmos. A validação funcional em Windows/Python 3.14 depende da execução local.

Foi realizada resolução sem instalação das 11 dependências diretas, exigindo wheels para CPython 3.14/Windows x86-64: todas foram encontradas. A resolução transitiva cruzada não foi concluída: o pip executado em Linux avaliou marcadores Linux do Torch apesar do alvo Windows. Portanto, essa checagem não equivale a instalação ou execução em Windows.
