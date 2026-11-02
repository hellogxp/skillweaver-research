"""Unified entry point for the three recovered core experiments."""
from __future__ import annotations

import json
from experiments.config import argument_parser, seed_everything


def main():
    parser = argument_parser(__doc__)
    parser.add_argument('experiment', choices=['main', 'convergence', 'transfer', 'all'])
    parser.add_argument('--n-queries', type=int, help='Optional main/convergence smoke subset; transfer uses the complete dataset.')
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--hints', type=int, default=15)
    parser.add_argument('--validate-only', action='store_true', help='Validate data/configuration without loading model weights.')
    args = parser.parse_args()
    if args.n_queries is not None and args.n_queries < 1:
        parser.error('--n-queries must be positive')
    if args.rounds < 1 or args.hints < 1:
        parser.error('--rounds and --hints must be positive')
    if args.n_queries and args.experiment in ['transfer', 'all']:
        parser.error('Do not subset the dataset when reproducing transfer partitions')

    from experiments import core_suite as suite
    from experiments.validate import validate
    validate(args.data_dir)
    suite.SKILL_POOL_PATH = args.data_dir / 'processed_v3/skill_pool.jsonl'
    suite.QUERIES_PATH = args.query_file or args.data_dir / 'benchmark_v3/compositional_queries.jsonl'
    suite.RESULTS_DIR = args.output_dir
    suite.H, suite.MAX_ROUNDS = args.hints, args.rounds
    suite.RANDOM_SEED = args.seed
    suite.EMBEDDING_MODEL = args.encoder
    seed_everything(args.seed)
    skills, queries = suite.load_skills(), suite.load_queries()
    if args.n_queries:
        queries = queries[:args.n_queries]
    config = dict(experiment=args.experiment, model=args.model, encoder=args.encoder,
                  seed=args.seed, hints=args.hints, rounds=args.rounds,
                  n_skills=len(skills), n_queries=len(queries))
    print(json.dumps(config, indent=2))
    if args.validate_only:
        return
    seed_everything(args.seed, include_torch=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    by_id = {s.skill_id:s for s in skills}
    retriever = suite.SkillRetriever(encoder_name=args.encoder, use_body=False, top_k=10)
    retriever.build_index(skills)
    decomposer = suite.LocalDecomposer(model_path=args.model, temperature=0.1, seed=args.seed)
    if args.experiment in ['main','all']:
        suite.run_experiment_1_main(decomposer,retriever,queries,by_id)
    if args.experiment in ['convergence','all']:
        suite.run_experiment_2_convergence(decomposer,retriever,queries,by_id)
    if args.experiment in ['transfer','all']:
        suite.run_experiment_3_transfer(decomposer,suite.SkillRetriever,skills,queries,by_id)


if __name__ == '__main__':
    main()

