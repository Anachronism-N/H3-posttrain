"""Deterministic one-to-one graph agreement; extractor-dependent, not music quality."""
import re
from collections import Counter

SUPPORTED_PARTS = frozenset(['arm','forearm','hand','wrist','elbow','knee'])


def key(node):
    return (node['body_part'],node['side'],re.sub(r'\s+',' ',node['action'].casefold()).strip(),node['direction'],node['negated'])


def prf(tp, proposed, expected):
    precision = tp / proposed if proposed else (1.0 if not expected else 0.0)
    recall = tp / expected if expected else (1.0 if not proposed else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp,proposed=proposed,expected=expected,precision=precision,recall=recall,f1=f1,
        empty_both=not proposed and not expected)


def relation_set(graph, included):
    before = {(e['source'],e['target']) for e in graph['edges'] if e['relation']=='before' and e['source'] in included and e['target'] in included}
    # Explicit transitive closure: reduces false differences from equivalent edge representations.
    while True:
        extra = {(a,d) for a,b in before for c,d in before if b==c}
        if extra <= before: break
        before |= extra
    cycles = sorted(a for a,b in before if a==b)
    relations = {(a,b,'before') for a,b in before if a!=b}
    for edge in graph['edges']:
        a,b=edge['source'],edge['target']
        if edge['relation']=='concurrent' and a in included and b in included:
            relations.add((min(a,b),max(a,b),'concurrent'))
    return relations,cycles


def evaluate_graph(reference, candidate):
    refs = [n for n in reference['nodes'] if n['body_part'] in SUPPORTED_PARTS]
    proposed = [n for n in candidate['nodes'] if n['body_part'] in SUPPORTED_PARTS]
    # Exact canonical tuples make the semantic boundary explicit. No fuzzy threshold tuned on test.
    # For identical repeated facts use textual occurrence order; ambiguity is reported separately.
    ref_groups = {}
    for node in refs: ref_groups.setdefault(key(node),[]).append(node['id'])
    mapping = {}; counts = Counter()
    for node in proposed:
        k=key(node);index=counts[k];counts[k]+=1
        if index<len(ref_groups.get(k,[])):mapping[node['id']]=ref_groups[k][index]
    facts = prf(len(mapping),len(proposed),len(refs))
    expected_rel,ref_cycles = relation_set(reference,{n['id'] for n in refs})
    raw_rel,cand_cycles = relation_set(candidate,{n['id'] for n in proposed})
    mapped_rel = set();unmatched_edges=0
    for a,b,relation in raw_rel:
        if a not in mapping or b not in mapping:unmatched_edges+=1;continue
        a,b=mapping[a],mapping[b]
        mapped_rel.add((min(a,b),max(a,b),relation) if relation=='concurrent' else (a,b,relation))
    relations = prf(len(mapped_rel & expected_rel),len(raw_rel),len(expected_rel))
    return dict(facts=facts,relations=relations,mapping=mapping,
        excluded_reference_nodes=len(reference['nodes'])-len(refs),excluded_candidate_nodes=len(candidate['nodes'])-len(proposed),
        unmatched_relation_edges=unmatched_edges,reference_temporal_cycles=ref_cycles,candidate_temporal_cycles=cand_cycles,
        repeated_fact_alignment_ambiguous=any(v>1 for v in Counter(key(n) for n in refs).values()),
        scope='exact canonical source-text agreement on supported arms/hands/elbows/knees; not unique choreography truth',
        direction_coordinate_truth_validated=False,model_assisted_extraction=True)
