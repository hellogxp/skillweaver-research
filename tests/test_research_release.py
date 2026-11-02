"""Offline regression checks for release packaging and experiment protocols."""
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from skillweaver.core.models import Skill, SubTask
from skillweaver.core.retriever import SkillRetriever

ROOT = Path(__file__).resolve().parent.parent


def test_frozen_data_references():
    from experiments.validate import validate
    report = validate(ROOT / 'data')
    assert (report['skills'],report['categories']) == (2209,24)
    assert report['datasets']['compositional_queries']['queries'] == 300
    assert report['datasets']['compositional_queries']['subtasks'] == 830
    assert report['datasets']['human_queries']['queries'] == 50


def test_encoder_failure_does_not_change_experiment(monkeypatch):
    attempts=[]
    def fail(name):
        attempts.append(name)
        raise OSError('model unavailable')
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=fail))
    r=SkillRetriever(encoder_name='requested-encoder')
    with pytest.raises(OSError):
        r._load_encoder()
    assert attempts == ['requested-encoder']


def test_real_faiss_with_controlled_embeddings():
    """Tests retrieval/index mechanics, not neural model quality."""
    class Encoder:
        def encode(self,texts,**kwargs):
            return np.array([[1.,0.] if 'alpha' in t else [0.,1.] for t in texts],dtype=np.float32)
    r=SkillRetriever(top_k=2)
    r._encoder=Encoder()
    r.build_index([Skill(skill_id='a',name='alpha',description='alpha'),
                   Skill(skill_id='b',name='beta',description='beta')])
    assert r.search('alpha')[0].skill.skill_id == 'a'
    assert r.search('beta')[0].skill.skill_id == 'b'
    assert r.search_batch([]) == []


def test_cpu_model_loading_is_explicit(monkeypatch):
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False),float32=object())
    monkeypatch.setitem(sys.modules,'torch',torch)
    from skillweaver.core.decomposer import LocalDecomposer
    kwargs=LocalDecomposer(model_path='unused',device='auto')._build_load_kwargs()
    assert kwargs['device_map'] == {'':'cpu'}
    assert kwargs['torch_dtype'] is torch.float32
    assert kwargs['trust_remote_code'] is False


def test_core_main_convergence_transfer_protocols(tmp_path):
    """Execute all three protocols with controlled components and real labels."""
    from experiments import core_suite as suite
    class Decomposer:
        def decompose(self,query):
            return [SubTask(step_index=0,description=query)]
        def decompose_with_hints(self,query,hints):
            return self.decompose(query)
    class Retriever:
        def __init__(self,**kwargs):pass
        def build_index(self,skills):pass
        def search_batch(self,texts):return [[] for _ in texts]
    skills=[Skill(skill_id=f'{c}-{i}',name=f'{c}-{i}',description='sample',categories=[c])
            for c in ['a','b','c'] for i in range(25)]
    queries=[dict(query_id=str(i),query='do task',difficulty='easy',num_skills=1,
                  subtasks=[dict(required_category=s.categories[0],ground_truth_skill_id=s.skill_id)])
             for i,s in enumerate(skills)]
    before=(suite.RESULTS_DIR,suite.MAX_ROUNDS)
    suite.RESULTS_DIR,suite.MAX_ROUNDS=tmp_path,2
    try:
        d,r=Decomposer(),Retriever()
        main=suite.run_experiment_1_main(d,r,queries,{})
        convergence=suite.run_experiment_2_convergence(d,r,queries,{})
        transfer=suite.run_experiment_3_transfer(d,Retriever,skills,queries,{})
    finally:
        suite.RESULTS_DIR,suite.MAX_ROUNDS=before
    assert len(main['sad']) == 75 and main['sad'][0]['da'] == 1
    assert len(convergence) == 2
    assert len(transfer['leave_2_categories_out']['held_out_categories']) == 2
    assert all(b['n_evaluated_queries']==len(b['sad']) for b in transfer.values())
    assert json.loads((tmp_path/'main_results_v4.json').read_text()) == main


@pytest.mark.parametrize('module',['compose','human_compose','human_style','compatibility','core_suite'])
def test_experiment_data_loaders(module):
    m=importlib.import_module('experiments.'+module)
    if hasattr(m,'load_skills'):
        assert len(m.load_skills()) == 2209
    elif hasattr(m,'load_data'):
        result=m.load_data()
        assert len(result[0]) == 2209


def test_reference_main_results_are_preserved():
    o=json.loads((ROOT/'results/reference/main_results_v4.json').read_text())
    assert sum(r['da'] for r in o['vanilla']) == 153
    assert sum(r['da'] for r in o['sad']) == 203


def test_hint_sensitivity_finishes_and_saves_per_query(tmp_path,monkeypatch):
    from experiments import hint_sensitivity as module
    from skillweaver.core.models import SkillMatch
    skill=Skill(skill_id='a',name='sample',description='sample',categories=['a'])
    class Decomposer:
        def __init__(self,**kwargs):pass
        def decompose(self,query):return [SubTask(step_index=0,description=query)]
        def decompose_with_hints(self,query,hints):return self.decompose(query)
    class Retriever:
        def __init__(self,**kwargs):pass
        def build_index(self,skills):pass
        def search_batch(self,texts):return [[SkillMatch(skill=skill,score=1.)] for _ in texts]
    monkeypatch.setattr(module,'LocalDecomposer',Decomposer)
    monkeypatch.setattr(module,'SkillRetriever',Retriever)
    monkeypatch.setattr(module,'RESULTS_DIR',tmp_path)
    monkeypatch.setattr(module,'load_skills',lambda:[skill])
    monkeypatch.setattr(module,'load_queries',lambda:[dict(query='sample',num_skills=1,subtasks=[dict(required_category='a')])])
    (tmp_path/'main_results_v4.json').write_text(json.dumps({'sad':[dict(da=1,cat_r1=1,cat_r10=1,chain_cat=1)]}))
    module.main()
    result=json.loads((tmp_path/'h_sensitivity_v4.json').read_text())
    assert set(result)=={'h5','h10','h15','h25'}
    assert len(result['h5']['per_query'])==1
