# Revisão técnica da iteração 2.5

Revisão do PR #3, realizada antes da validação na máquina do usuário e do merge.

## Achados corrigidos

1. **Elitismo total na seleção por rank:** a API aceitava fração elitista 1,0, mas podia tentar amostrar de um conjunto vazio quando todos os candidatos de um nicho já estavam garantidos. Agora retorna diretamente os indivíduos elitistas quando a cota já está preenchida. Um teste cobre o limite e verifica igualdade com a seleção determinística. O experimento full usou fração 0,1; esse caso não afetou os resultados publicados.
2. **Auditorias com `assert`:** o modo `python -O` removia as verificações. Os dois scripts de auditoria agora usam condições e `ValueError`, ativas também no modo otimizado. Um teste apresenta uma rodada incompleta sob `-O` e exige rejeição explícita, sem gerar aprovação.

## Verificações concluídas

- 28 testes passaram, incluindo os dois novos casos de regressão.
- Auditoria dos 725 registros completos reaplicada com `python -O`: hashes dos jobs, índices disjuntos e perdas finitas aprovados.
- Reconstrução de folds, seeds, pareamento e decisões internas reaplicada com `python -O`: aprovada.
- Todos os arquivos produzidos novamente pelo consolidador coincidiram byte a byte com a consolidação versionada. As métricas e conclusões permanecem iguais.
- Revisão do fluxo: ajuste/calibração usam treino; triagem e confirmação usam subdivisões internas; teste externo é posterior à decisão. Utilidade e controles externos não escolhem hiperparâmetros.
- Escore mantém 50% candidatos e 50% sobreviventes; a Gaussiana direta é explicitamente reportada como amostra final. A diferença entre 150 sintéticos de treino e 120 reais de treino está documentada na avaliação de utilidade.

## Proveniência e reprodução

A execução científica original corresponde ao commit remoto `bb41d38661b159f7f1ae094ad73ed13c470db45f`. Seus manifestos, resultados e arquivo bruto permanecem intactos. O registro `code_identity.json` descreve a verificação feita naquele commit, antes desta revisão; não afirma identidade dos arquivos corrigidos com o manifesto histórico.

As correções alteram hashes de código. Para retomar/reproduzir exatamente aquela execução, use o commit original e o ambiente registrado. Uma execução com o código revisado deve usar uma nova pasta, como exige o mecanismo de manifesto. Não houve reexecução dos 725 experimentos: os casos corrigidos não mudam o caminho usado naquela avaliação.

## Situação

Revisão técnica e conferência dos resultados concluídas com as correções acima. A avaliação continua exploratória na Iris, sem ganho adicional demonstrado sobre a 2.4. Ainda faltam a validação local do usuário e sua autorização para o merge. Nenhuma mudança nos defaults de geração.
