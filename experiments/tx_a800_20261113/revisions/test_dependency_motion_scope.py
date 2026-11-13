import unittest
import en_core_web_sm
from captionlab.dependency_motion_graph import extract


class DependencyScopeTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.nlp=en_core_web_sm.load(disable=['ner'])
 def graph(self,text):return extract(self.nlp(text))['graph']
 def test_gaze_and_chest_do_not_inherit_arm(self):
  graph=self.graph('Raise one arm, turn their gaze to the side, and then swing their arm outward.')
  self.assertNotIn('rotate',[n['action'] for n in graph['nodes']])
  graph=self.graph('A quick arm chop and chest thrust accent the last hit.')
  self.assertEqual([n['action'] for n in graph['nodes']],['chop'])
 def test_return_direction_is_not_backward(self):
  graph=self.graph('The left hand swings left and back right.')
  self.assertFalse(any(n['direction']=='backward' for n in graph['nodes']))
 def test_lift_modifier_does_not_change_bend_direction(self):
  graph=self.graph('The other arm bends into a rightward lift.')
  bends=[n for n in graph['nodes'] if n['action']=='bend']
  self.assertTrue(bends)
  self.assertTrue(all(n['direction']=='unspecified' for n in bends))
 def test_knee_state_is_not_generic_pose(self):
  graph=self.graph('Hold a bent-knee stance with arms outstretched.')
  self.assertTrue(any(n['body_part']=='knee' and n['action']=='hold bent' for n in graph['nodes']))
 def test_real_conjoined_motion_and_order_retained(self):
  graph=self.graph('Move the arms forward and then lower them.')
  self.assertEqual([n['action'] for n in graph['nodes']],['move','lower'])
  self.assertTrue(any(e['relation']=='before' for e in graph['edges']))
 def test_direction_not_anatomical_side(self):
  graph=self.graph('Both hands move leftward while the right knee bends.')
  hands=[n for n in graph['nodes'] if n['body_part']=='hand']
  self.assertTrue(hands)
  self.assertTrue(all(n['side']=='both' for n in hands))
  self.assertTrue(any(n['direction']=='left' for n in hands))


if __name__=='__main__':unittest.main()
