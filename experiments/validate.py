"""Offline validation of the frozen benchmark and available legacy query set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(data_dir):
    data_dir = Path(data_dir)
    skills = read_jsonl(data_dir / 'processed_v3/skill_pool.jsonl')
    by_id = {s['skill_id']:s for s in skills}
    if len(skills) != len(by_id):
        raise ValueError('Duplicate skill IDs')
    categories = {c for s in skills for c in s['categories']}
    report = {'skills':len(skills), 'categories':len(categories), 'datasets':{}}
    for name in ['compositional_queries','human_queries']:
        queries = read_jsonl(data_dir / 'benchmark_v3' / (name + '.jsonl'))
        if len({q['query_id'] for q in queries}) != len(queries):
            raise ValueError('Duplicate query IDs: ' + name)
        for q in queries:
            if q['num_skills'] != len(q['subtasks']):
                raise ValueError('Ground-truth count mismatch: ' + q['query_id'])
            for step in q['subtasks']:
                skill = by_id.get(step['ground_truth_skill_id'])
                if skill is None or step['required_category'] not in skill['categories']:
                    raise ValueError('Invalid ground-truth reference: ' + q['query_id'])
        report['datasets'][name] = {
            'queries':len(queries),
            'subtasks':sum(len(q['subtasks']) for q in queries),
            'difficulty':dict(Counter(q['difficulty'] for q in queries)),
        }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent.parent / 'data')
    print(json.dumps(validate(parser.parse_args().data_dir),indent=2))

