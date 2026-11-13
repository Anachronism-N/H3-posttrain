import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from choreobench.fixtures import create_demo
from choreobench.audio import make_windows
from choreobench.planning_units import build_planning_units
from choreobench.io import read_json,file_sha
from choreobench.composition_workflows import run_composition


class FixtureProvider:
    identity={'kind':'replay','synthetic':True,'purpose':'CPU-only contract test, not experiment'}
    def __init__(self):self.requests=[]
    def generate(self,request):
        self.requests.append(request)
        data=json.loads(request['messages'][-1]['content'])
        if request['response_kind']=='coordination_outline':
            result=dict(motifs=[],sections=[dict(section_id=s['section_id'],intent='A simple arm phrase.',motif_ids=[]) for s in data['song']['music']['sections']])
        elif request['response_kind']=='composition_feedback':result=dict(feedback='Keep the current simple phrase; no new evidence calls.')
        else:
            proposal=None
            if 'planning_unit' in data:
                schedule=data['event_clock']['permitted_schedules'][0]
                event=dict(schedule[0],track='body',action='Raise both arms gently, then lower them while softly bending the knees.',motif_id=None,source_call_ids=[data['local_proposals'][0]['call_id']])
                proposal=dict(unit_id=data['planning_unit']['unit_id'],events=[event])
            if 'composition_propose'==request['response_kind'] and 'tool' not in request['response_schema']['properties'] or request['response_kind']=='composition_revise':result=proposal
            else:
                observations=data['tool_observations']
                if data['committed_units']==data['total_units']:tool='finish'
                elif data['current_draft'] is None:tool='propose_block'
                elif observations and observations[-1].get('tool')=='check_block':tool='commit_block'
                else:tool='check_block'
                result=dict(tool=tool,proposal=proposal if tool=='propose_block' else None,reason='Synthetic test decision')
        return dict(text=json.dumps(result),raw=dict(fixture=True),identity=self.identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=100,output_tokens=100,latency_seconds=0))


class CompositionRuntimeTest(unittest.TestCase):
    def test_all_methods_causal_commit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_demo(root/'fixture',12);dataset=read_json(root/'fixture/dataset.json');song=dataset['songs'][0]
            def source(path,song_id,cfg,sources):
                sources[str(root/'fixture/dataset.json')]=file_sha(root/'fixture/dataset.json')
                windows=make_windows(song['audio']['num_samples'],song['audio']['sample_rate'])
                options=dict(windows=[dict(window=w,candidates=[dict(call_id='fixture/'+str(w['index']),raw_text='Raise both arms gently.')]) for w in windows])
                return dict(original=song,song=copy.deepcopy(song),options=options,units=build_planning_units(song,windows,cfg['unit_config']),config=cfg,synthetic=True,source_contract_sha256='fixture')
            for method in ('workflow','self_refine','react','choreography'):
                provider=FixtureProvider();output=root/method
                with patch('choreobench.composition_workflows._source',source),patch('choreobench.composition_workflows.create_provider',return_value=provider):
                    result=run_composition(root/'fixture/dataset.json',song['song_id'],dict(kind='replay'),output,method=method)
                    self.assertTrue(result['complete']);self.assertTrue(result['full_song_succeeded'],result)
                    self.assertEqual(result['committed_units'],result['total_units'])
                    before=len(provider.requests)
                    replay=run_composition(root/'fixture/dataset.json',song['song_id'],dict(kind='replay'),output,method=method,resume=True)
                    self.assertEqual(result,replay);self.assertEqual(before,len(provider.requests))
                plan=read_json(output/'plan.json');self.assertEqual(plan['events'][0]['start'],0)
                self.assertEqual(plan['events'][-1]['end'],song['audio']['num_samples'])
                if method in ('react','choreography'):
                    for request in provider.requests[1:]:
                        self.assertIn('remaining_model_calls',json.loads(request['messages'][-1]['content']))
                self.assertFalse(result['quality_verified']);self.assertFalse(result['formal_E3_completed'])


if __name__=='__main__':unittest.main()
