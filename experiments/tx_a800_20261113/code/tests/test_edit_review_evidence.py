import copy,json,unittest
from captionlab.edit_review_evidence import parse_review,review_schema
from captionlab.e4_final_metrics import SCHEMA,FIELDS

class EditReviewEvidenceTest(unittest.TestCase):
 def setUp(self):
  self.payload=dict(target_ids=['target'],final_events=[dict(event_id='target',action='Raise both arms while keeping both knees bent.'),dict(event_id='next',action='Lower both arms, then straighten both knees.')])
  self.facts=[dict(fact_id='f0',description='Both knees remain bent')]
  self.rating=dict(goal_complete='yes',protected_facts_preserved='yes',no_new_boundary_conflict='yes',before_needs_change='yes',reason='Supported',goal_quote='Raise both arms',fact_reviews=[dict(fact_id='f0',status='retained',quote='keeping both knees bent')],boundary_conflict_quotes=[])
 def parse(self,rating):return parse_review(json.dumps(rating),self.payload,self.facts,FIELDS,SCHEMA)
 def test_valid_evidence_and_omission(self):
  self.assertEqual(self.parse(self.rating)['protected_facts_preserved'],'yes')
  self.rating['fact_reviews'][0].update(status='omitted',quote='')
  self.assertEqual(self.parse(self.rating)['protected_facts_preserved'],'no')
 def test_unknown_fact_does_not_turn_into_retention(self):
  self.rating['fact_reviews'][0].update(status='unknown',quote='')
  self.assertEqual(self.parse(self.rating)['protected_facts_preserved'],'unknown')
 def test_neighbor_cannot_prove_target_goal(self):
  self.rating['goal_quote']='straighten both knees'
  with self.assertRaises(ValueError):self.parse(self.rating)
 def test_goal_and_fact_require_exact_text(self):
  self.rating['goal_quote']='invented movement'
  with self.assertRaises(ValueError):self.parse(self.rating)
  self.rating['goal_quote']='Raise both arms';self.rating['fact_reviews'][0]['quote']='invented fact'
  with self.assertRaises(ValueError):self.parse(self.rating)
 def test_conflict_requires_two_distinct_clauses(self):
  self.rating['no_new_boundary_conflict']='no';self.rating['boundary_conflict_quotes']=['Raise both arms']
  with self.assertRaises(ValueError):self.parse(self.rating)
 def test_missing_and_duplicate_fact_rejected(self):
  self.rating['fact_reviews']=[]
  with self.assertRaises(ValueError):self.parse(self.rating)
 def test_review_schema_preserves_input(self):
  old=copy.deepcopy(SCHEMA);review_schema(SCHEMA,self.facts);self.assertEqual(SCHEMA,old)

if __name__=='__main__':unittest.main()
