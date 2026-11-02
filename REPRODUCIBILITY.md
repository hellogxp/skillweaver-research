# Reproducibility notes

## Inputs and metrics

- Main benchmark: 300 queries, 830 ordered sub-tasks; 150 easy / 100 medium / 50 hard.
- Catalog: 2,209 unique skill IDs across 24 categories. Every annotated skill ID and category resolves in this snapshot. Skill bodies are empty.
- Legacy human-style input: 50 queries, 145 sub-tasks; 20 easy / 20 medium / 10 hard. Human-style denotes LLM-generated natural-language queries, not human participant data.
- The paper's separate 200-query input/annotation file is unavailable. Preserved metric arrays cannot reconstruct its queries, labels, or missing predictions. No replacement queries were generated for this release.

DA measures exact predicted/reference step-count agreement. CatR@k aligns predicted and annotated sub-tasks by position and checks for the reference category in the first k retrieved skills. It is not exact skill-ID accuracy. Compose edge references are sequential-chain proxies from ordered annotations.

## Reference results

`results/reference/` preserves selected historical JSON artifacts, not newly executed neural results. The main reference contains 153/300 vanilla and 203/300 SAD step-count matches. The held-out-skill reference lists 139 eligible queries but evaluates 100; the category-held-out reference evaluates 62. Current scripts record eligibility and evaluated counts separately.

Some historical artifacts contain only aggregates or omit query IDs and exact model/software revisions. The qwen-max baseline reference lacks a complete corresponding inference runner here. Original stochastic generation streams and exact historical environments have not been reconstructed.

The supplied main protocol uses temperature 0.1, top-k 10, H=15, and seed 42. Do not assume these reproduce every setting described elsewhere in the paper or submission checklist. The 830-versus-828 sub-task discrepancy remains unresolved; labels have not been modified to force agreement.

## Release implementation

Paths and experiment outputs are configurable. The retriever uses real FAISS and fails if the requested encoder cannot load. Controlled test embeddings validate indexing mechanics; they do not measure neural retrieval quality. Local model loading disables remote model code execution for the built-in architectures. Historical reference results remain separate from new `runs/` outputs.

Offline tests, data validation, and the minimal example do not constitute a rerun of Qwen/MiniLM inference. Use a compatible isolated model environment for full experiments. One development environment had a native FAISS/Torch library import conflict; no environment override was used to suppress it.

One catalog source URL was normalized from a security-proxy host to its direct public GitHub URL. IDs, descriptions, annotations, row order, and numerical reference results were preserved.
