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
