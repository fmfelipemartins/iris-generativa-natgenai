# Referências experimentais

## Refinamentos atuais

- `refinement_2_4_summary/`: consolidação versionada da avaliação 2.4, lida pela seção 7 do notebook.
- `refinement_2_5_summary/`: consolidação versionada de 725 execuções da 2.5, lida pela seção 8; inclui métricas, seleção interna, controles, utilidade, motivos de rejeição e comparação com a 2.4.
- `refinement_2_5_diagnosis/`: diagnósticos e verificações da entrega original; registros históricos não são substituídos pelos testes posteriores.
- Os jobs brutos da 2.5 estão preservados em `iris_natgenai_refinamento_2_5_evidencias.zip`, fora desta consolidação Git. SHA-256: `d6fad133ffb1ac2139fb83c1d812a573ca5b199bbc3a5f5a20aa3a2e72e8836b`.

Consultar tabelas salvas não exige repetir o perfil full. Novas execuções e consolidações devem usar pastas próprias para preservar as referências. Os hashes históricos descrevem o código original, anterior à revisão técnica. Ver [proveniência](../docs/REVISAO_2_5.md), [resultados 2.4](../docs/RESULTADOS_REFINAMENTO_2_4.md) e [resultados 2.5](../docs/RESULTADOS_REFINAMENTO_2_5.md).

A perda conjunta combina candidatos e sobreviventes; a Gaussiana direta tem somente perda de amostra final. A acurácia de classificação de espécies mede utilidade e não deve ser confundida com a acurácia do discriminador real/sintético.

## Referências históricas

- `baseline_full_20261019/`: notebook completo enviado pelo usuário, preservado byte a byte, com checksum e informações de origem. Não contém a correção SBX desta revisão.
- `review_full_20261019/`: resultados da revisão 2.3.1 em orçamento completo nos protocolos afetados. Os metadados registram configuração e versões por fase; marcadores `COMPLETE_*` indicam conclusão dos trabalhos daquela fase.
- `execution_<perfil>_<data>/`: exportações locais produzidas ao executar o notebook; não versionadas automaticamente. Cada execução cria uma nova pasta.

Na revisão, `sbx_paired.csv` contém valores por seed/modo/condição; `sbx_summary.csv` é sua agregação. Os arquivos `vae_<variante>_seed<n>_*` guardam perdas por época e medições por geração/estágio/tarefa; `vae_ablation.csv` separa candidatos da última geração e sobreviventes. `founder_fraction` mede retenção de indivíduos iniciais. Os valores geométricos dos diagnósticos usam unidades padronizadas; `accuracy` e `auc` pertencem aos discriminadores. Contagens de avaliações de fitness excluem sondagens observacionais, mas tempos incluem seu custo.

`crossfit_corrected.csv` preserva índices de treino/holdout, enquanto os demais CSVs documentam sensibilidade, ablação, mutação e controles 2D. As figuras `benchmark_0.png` e `benchmark_1.png` correspondem aos resultados 2D completos. A ausência de superioridade geral ou de um VAE com diversidade adequada é parte do resultado, não uma falha omitida da validação.

Ver [relatório metodológico](../docs/REVISAO_METODOLOGICA.md) para escopo, ambiente, definições e comandos de reprodução. Use uma nova pasta `--output` para não sobrescrever estas referências.
