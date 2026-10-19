# Referências experimentais

- `baseline_full_20261019/`: notebook completo enviado pelo usuário, preservado byte a byte, com checksum e informações de origem. Não contém a correção SBX desta revisão.
- `review_full_20261019/`: resultados da revisão 2.3.1 em orçamento completo nos protocolos afetados. Os metadados registram configuração e versões por fase; marcadores `COMPLETE_*` indicam conclusão dos trabalhos daquela fase.
- `execution_<perfil>_<data>/`: exportações locais produzidas ao executar o notebook; não versionadas automaticamente. Cada execução cria uma nova pasta.

Na revisão, `sbx_paired.csv` contém valores por seed/modo/condição; `sbx_summary.csv` é sua agregação. Os arquivos `vae_<variante>_seed<n>_*` guardam perdas por época e medições por geração/estágio/tarefa; `vae_ablation.csv` separa candidatos da última geração e sobreviventes. `founder_fraction` mede retenção de indivíduos iniciais. Os valores geométricos dos diagnósticos usam unidades padronizadas; `accuracy` e `auc` pertencem aos discriminadores. Contagens de avaliações de fitness excluem sondagens observacionais, mas tempos incluem seu custo.

`crossfit_corrected.csv` preserva índices de treino/holdout, enquanto os demais CSVs documentam sensibilidade, ablação, mutação e controles 2D. As figuras `benchmark_0.png` e `benchmark_1.png` correspondem aos resultados 2D completos. A ausência de superioridade geral ou de um VAE com diversidade adequada é parte do resultado, não uma falha omitida da validação.

Ver [relatório metodológico](../docs/REVISAO_METODOLOGICA.md) para escopo, ambiente, definições e comandos de reprodução. Use uma nova pasta `--output` para não sobrescrever estas referências.
