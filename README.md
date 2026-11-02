# SkillWeaver research implementation

Code, CompSkillBench data, and evaluation scripts for **Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compose**.

[Paper](https://arxiv.org/abs/2606.18051) · [Data](data/README.md) · [Experiment notes](REPRODUCIBILITY.md)

SkillWeaver decomposes a task, retrieves candidate skills, aligns the decomposition with the catalog through Skill-Aware Decomposition (SAD), and constructs a skill plan. This research package produces plans; it does not execute external tools.

## Included

- Core decomposition, retrieval, SAD, compatibility, and planning implementations.
- CompSkillBench: 300 queries and 2,209 skill metadata records across 24 categories.
- A separate 50-query legacy human-style set.
- Main, convergence, transfer, composition, and supplementary experiment scripts.
- Selected historical reference results and offline tests.

## Install and check

Use Python 3.10 or newer in an isolated environment.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m experiments.validate
python examples/minimal.py
python -m pytest -q
```

These checks do not download model weights or call a paid API. Tests marked `model` require an installed neural encoder and its weights; run them explicitly with `python -m pytest -o addopts='' -m model -q` after installing model dependencies.

## Run experiments

```sh
python -m pip install -e '.[models,analysis]'
python -m experiments.run main --validate-only
python -m experiments.run main --model /path/to/Qwen2.5-7B-Instruct --encoder /path/to/all-MiniLM-L6-v2
python -m experiments.run convergence --model /path/to/Qwen2.5-7B-Instruct --encoder /path/to/all-MiniLM-L6-v2
python -m experiments.run transfer --model /path/to/Qwen2.5-7B-Instruct --encoder /path/to/all-MiniLM-L6-v2
```

The default model identifiers are `Qwen/Qwen2.5-7B-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`. Obtain weights under their respective terms. `--data-dir`, `--output-dir`, `--seed`, `--hints`, and `--rounds` are configurable; new results default to `runs/`. Use `--n-queries 2` for a small main/convergence model smoke run. `all` runs main, convergence, and transfer.

Additional modules run with `python -m experiments.<module> --help`:

| Module | Purpose |
| --- | --- |
| `compose` | Sequential versus dependency-aware planning; saves SAD sub-task texts |
| `encoder_baselines` | TF-IDF, MiniLM, and Qwen embedding comparison |
| `sparse_baseline` | TF-IDF retrieval using saved SAD texts |
| `retrieval_ablation` | Retrieval configuration comparisons |
| `compatibility` | LLM-judged compatibility evaluation |
| `hint_sensitivity` | Hint-count ablation |
| `reranking` | Listwise reranking on 300 queries |
| `step_count_controls` | Oracle step-count control |
| `robustness` | Cross-model and paraphrase checks |
| `human_style`, `human_compose` | Evaluation on the separate 50-query set |

Run `compose` before encoder ablations and `main` before hint sensitivity, using the same output directory. Modules that require saved SAD texts must use actual saved predictions. To reuse historical texts, explicitly copy `results/reference/sad_subtask_texts.json` to the output directory and record that choice; this does not constitute a new decomposition run.

## Scope and limitations

The original input/annotation file for the paper's 200-query human-style evaluation is unavailable. The 50-query legacy set is different and must not be presented as that evaluation. The main dataset has 830 annotated sub-tasks; a paper passage reports 828. Historical model/software revisions are incomplete, and full neural experiments have not been rerun for this release. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

All catalog skill bodies are empty. The data supports metadata-based routing; it cannot establish a benefit from full-body retrieval. Compose edge labels are derived from ordered chains and are not independently annotated arbitrary DAGs.

## License and citation

The software and authored query/annotation files use [Apache-2.0](LICENSE). Catalog material compiled from the MIT-licensed awesome-mcp-servers collection retains its [upstream notice](data/UPSTREAM_LICENSE.txt). Referenced servers, products, and model weights retain their own terms; their code and weights are not included. See [data licensing and attribution](data/LICENSE.md).

Please cite the paper when using the implementation or benchmark; machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
