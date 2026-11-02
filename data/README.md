# CompSkillBench snapshot

| File | Records | Role |
| --- | ---: | --- |
| `benchmark_v3/compositional_queries.jsonl` | 300 | Main benchmark with ordered sub-task annotations |
| `processed_v3/skill_pool.jsonl` | 2,209 | Skill metadata catalog across 24 categories |
| `processed_v3/skill_pool_stats.json` | — | Catalog statistics |
| `benchmark_v3/human_queries.jsonl` | 50 | Separate legacy human-style set |

Queries contain `query_id`, query text, difficulty, `num_skills`, and ordered `subtasks`. Each annotated sub-task supplies its description, category, and ground-truth skill ID/name. The catalog supplies names, descriptions, categories, source URLs, and related public metadata; bodies are empty.

The catalog statistics identify [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) as the collection source. Public source locations remain in each record's `source_path` and, when available, `metadata.github_repo`. One proxy URL was normalized to its direct public location. No source server implementation, private environment configuration, or model weights are included.

See [LICENSE.md](LICENSE.md) for licensing and attribution, and the root experiment notes for dataset limitations. The 50-query file is not the paper's 200-query evaluation input.
