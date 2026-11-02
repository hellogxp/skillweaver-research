"""A local planning example without model weights, network requests or tool execution."""
from skillweaver.core.dag_planner import DAGPlanner
from skillweaver.core.models import Skill, SkillMatch, SubTask

skills = [Skill(skill_id='fetch',name='fetch',description='Fetch public example data'),
          Skill(skill_id='summarize',name='summarize',description='Summarize fetched data')]
plan = DAGPlanner().plan(
    query='fetch data and then summarize it',
    subtasks=[SubTask(step_index=0,description='fetch data'),
              SubTask(step_index=1,description='then summarize the fetched data')],
    candidates_per_step=[[SkillMatch(skill=s,score=1.0)] for s in skills],
)
print('Steps:',plan.num_steps,'Edges:',plan.edges)
