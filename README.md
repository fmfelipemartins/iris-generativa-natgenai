# Iris Generativa com NatGenAI

Experimento didático e de pesquisa sobre geração de dados sintéticos por Computação Evolutiva, utilizando a base **Iris** como problema controlado para estudo de operadores evolutivos, fidelidade estatística, diversidade e distinguibilidade entre dados reais e sintéticos.

O projeto foi estruturado como um pequeno pacote Python para separar a implementação do algoritmo da narrativa experimental apresentada no notebook principal.

## Objetivo

O experimento parte dos 150 registros reais da base Iris e gera 150 registros sintéticos, preservando a estrutura das três espécies:

- *Iris setosa*;
- *Iris versicolor*;
- *Iris virginica*.

A qualidade dos dados sintéticos é analisada por diferentes perspectivas, incluindo:

- plausibilidade;
- novidade;
- coerência com a espécie;
- diversidade;
- fidelidade marginal;
- duplicação exata;
- distinguibilidade entre dados reais e sintéticos.

Como teste discriminativo, os dados são reunidos em uma base com:

- `1 = real`;
- `0 = sintético`.

São avaliados classificadores lineares e não lineares para verificar em que medida a origem dos registros ainda pode ser identificada.

## Fundamentação experimental

O projeto investiga conceitos discutidos no trabalho **Evolutionary Computation as Natural Generative AI**, explorando Computação Evolutiva como mecanismo de geração de novas amostras.

Três condições principais são comparadas:

1. **SBX** — recombinação probabilística centrada nos pais;
2. **OB-Scan** — recombinação disruptiva baseada em densidade estimada por KDE;
3. **Híbrido 50/50** — combinação de SBX e OB-Scan.

A base Iris é tratada como um problema multitarefa, em que cada espécie corresponde a uma tarefa.

O fitness generativo é definido por:

$$
F(x)=0{,}50P(x)+0{,}25N(x)+0{,}25C(x)
$$

em que:

- `P(x)` representa plausibilidade;
- `N(x)` representa novidade;
- `C(x)` representa coerência com a tarefa/espécie.

## Estrutura do projeto

```text
iris-generativa-natgenai/
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

### Responsabilidades dos módulos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Hiperparâmetros, seeds e controles de execução |
| `data.py` | Carregamento da Iris, padronização, calibração e contexto evolutivo |
| `fitness.py` | Plausibilidade, novidade, coerência e fitness generativo |
| `operators.py` | SBX, OB-Scan-KDE e suporte aos operadores evolutivos |
| `evolution.py` | Inicialização populacional, seleção de pais, RMP, seleção moderada e ciclo evolutivo |
| `evaluation.py` | Métricas generativas, fidelidade estatística e classificadores Real × Sintético |
| `experiments.py` | Experimento principal, robustez, sensibilidade, mutação polinomial e cross-fitting |
| `notebooks/` | Narrativa experimental, execução, tabelas, gráficos e interpretação |

## Requisitos

O projeto requer **Python 3.11 ou superior**.

As principais dependências são:

- NumPy;
- pandas;
- Matplotlib;
- SciPy;
- scikit-learn;
- PyGAD;
- Jupyter;
- ipykernel.

As versões utilizadas no ambiente experimental estão registradas em `requirements.txt` e `pyproject.toml`.

## Instalação

### Windows + PowerShell

Na pasta raiz do projeto:

```powershell
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atualize o `pip`:

```powershell
python -m pip install --upgrade pip
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Instale o pacote local em modo editável:

```powershell
python -m pip install -e . --no-deps
```

Teste a instalação:

```powershell
python -c "import iris_generativa; print('Pacote instalado corretamente')"
```

O modo editável (`-e`) permite modificar os módulos em `iris_generativa/` sem reinstalar o pacote após cada alteração.

## Execução

Abra a pasta raiz do projeto no VS Code e, em seguida, abra:

```text
notebooks/Iris_Generativa_NatGenAI.ipynb
```

Selecione como kernel o Python do ambiente `.venv`.

Para uma execução limpa e reprodutível:

1. reinicie o kernel;
2. execute **Run All**;
3. aguarde a conclusão de todas as células;
4. salve o notebook.

## Experimentos

Além da comparação principal entre SBX, OB-Scan e a condição híbrida, o projeto inclui experimentos complementares.

### Robustez

Repete as condições principais em múltiplas seeds para avaliar estabilidade dos resultados.

### Sensibilidade do SBX

Avalia diferentes valores do parâmetro $\eta$, responsável pela concentração dos descendentes em torno dos pais.

As configurações estudadas incluem:

- $\eta = 5$;
- $\eta = 30$;
- $\eta = 50$.

### Random Mating Probability — RMP

Compara configurações com e sem transferência intertarefas, permitindo avaliar o efeito do cruzamento entre indivíduos associados a espécies diferentes.

### Seleção moderada

Avalia o efeito da preservação de nichos sobre diversidade populacional e distinguibilidade dos sintéticos.

### SBX + Polynomial Mutation

Experimenta a combinação entre SBX e mutação polinomial separadamente da comparação principal, evitando introduzir esse fator como confundidor no experimento SBX × OB-Scan.

### Cross-fitting

O protocolo cross-fitted separa os dados reais utilizados como referência pelo gerador daqueles utilizados posteriormente na avaliação.

Assim, os sintéticos são comparados com registros reais que não participaram de sua geração, fornecendo uma avaliação mais rigorosa de generalização.

## Métricas

O projeto utiliza diferentes métricas porque nenhuma medida isolada é suficiente para avaliar um gerador sintético.

Entre elas:

- número de duplicatas exatas;
- coerência por espécie;
- novidade média;
- diversidade;
- razão entre diversidade sintética e real;
- distância de Wasserstein;
- estatística de Kolmogorov-Smirnov;
- acurácia de Regressão Logística;
- acurácia de Random Forest;
- ROC-AUC;
- validação cruzada;
- avaliação cross-fitted.

A interpretação deve considerar conjuntamente:

$$
\text{fidelidade}
+
\text{novidade}
+
\text{diversidade}
+
\text{coerência}
$$

Baixa distinguibilidade Real × Sintético, isoladamente, não é suficiente para caracterizar um bom gerador, pois memorizar ou copiar registros reais também poderia reduzir a capacidade de discriminação.

## Reprodutibilidade

As seeds e os principais hiperparâmetros estão centralizados em:

```text
iris_generativa/config.py
```

A separação entre configuração, implementação e notebook busca facilitar:

- repetição dos experimentos;
- comparação entre condições;
- auditoria dos hiperparâmetros;
- extensão futura do projeto;
- reutilização dos módulos em outras bases.

## Organização para pesquisa

O notebook principal contém prioritariamente:

- contexto e perguntas experimentais;
- configuração;
- execução;
- tabelas;
- gráficos;
- interpretação.

A implementação de baixo nível permanece nos módulos Python do pacote `iris_generativa`.

Essa separação permite que o notebook funcione como documento de pesquisa, enquanto o pacote concentra a implementação reutilizável.

## Estado do projeto

O projeto encontra-se em desenvolvimento e utiliza a base Iris como ambiente controlado para estudo metodológico.

Mudanças futuras em fitness, operadores, seleção, hiperparâmetros, avaliação ou protocolo experimental devem ser tratadas como alterações metodológicas e documentadas separadamente de refatorações estruturais.

## Referências principais

- Shi, Y. et al. **Evolutionary Computation as Natural Generative AI**. arXiv:2510.08590.
- Whitley, D. **A Genetic Algorithm Tutorial**. *Statistics and Computing*, 1994.
- Mitchell, M. **An Introduction to Genetic Algorithms**. MIT Press, 1996.
- Luke, S. **Essentials of Metaheuristics**.
- PyGAD — documentação oficial.

## Licença

Ainda não foi definida uma licença para o repositório.