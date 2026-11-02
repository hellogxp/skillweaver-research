"""Run the TF-IDF baseline with explicitly selected saved decomposition texts."""
import json
from experiments.config import ROOT, argument_parser


def main():
    parser=argument_parser(__doc__)
    parser.add_argument('--sad-texts',type=type(ROOT),default=ROOT/'results/reference/sad_subtask_texts.json')
    args=parser.parse_args()
    from experiments import encoder_baselines as module
    from experiments.validate import read_jsonl
    from skillweaver.core.models import Skill
    skills=[Skill.from_dict(o) for o in read_jsonl(args.data_dir/'processed_v3/skill_pool.jsonl')]
    queries=read_jsonl(args.query_file or args.data_dir/'benchmark_v3/compositional_queries.jsonl')
    texts={}
    for item in json.loads(args.sad_texts.read_text()):
        texts.setdefault(item['query_id'],[]).append(item['subtask_text'])
    missing=[q['query_id'] for q in queries if q['query_id'] not in texts]
    if missing:
        parser.error(f'Saved decompositions missing for {len(missing)} queries; ground-truth substitution is disabled')
    result=module.run_tfidf(skills,queries,texts)
    result['decomposition_source']=str(args.sad_texts)
    result['new_neural_inference']=False
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'tfidf_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_query'},indent=2))


if __name__=='__main__':
    main()
