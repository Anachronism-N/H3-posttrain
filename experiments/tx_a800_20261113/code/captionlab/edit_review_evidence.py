"""Evidence-bound edit reviews. Model semantics stay distinct from literal checks."""
from choreobench.providers import parse_json_response


def review_schema(base,facts):
    import copy
    schema=copy.deepcopy(base)
    schema['required']+=['fact_reviews','goal_quote','boundary_conflict_quotes']
    schema['properties']['goal_quote']=dict(type='string',maxLength=500)
    schema['properties']['boundary_conflict_quotes']=dict(type='array',maxItems=4,items=dict(type='string',minLength=1,maxLength=500))
    fact=dict(type='object',additionalProperties=False,required=['fact_id','status','quote'],properties=dict(
        fact_id=dict(type='string',enum=[x['fact_id'] for x in facts] or ['none']),
        status=dict(type='string',enum=['retained','omitted','contradicted','unknown']),quote=dict(type='string',maxLength=500)))
    schema['properties']['fact_reviews']=dict(type='array',minItems=len(facts),maxItems=len(facts),items=fact)
    return schema


def parse_review(raw,payload,facts,fields,base):
    rating=parse_json_response(raw)
    if set(rating)!=set(review_schema(base,facts)['required']) or any(rating[k] not in ('yes','no','unknown') for k in fields):
        raise ValueError('Edit review schema/labels differ')
    final=chr(10).join(e['action'] for e in payload['final_events'])
    target_ids=set(payload.get('target_ids',[]))
    target=chr(10).join(e['action'] for e in payload['final_events'] if not target_ids or e['event_id'] in target_ids)
    targets=chr(10).join(e['action'] for e in payload['final_events'] if e['event_id'] in payload['target_ids'])
    if not isinstance(rating['reason'],str) or not rating['reason'].strip():raise ValueError('Missing review reason')
    quote=rating['goal_quote']
    if not isinstance(quote,str) or quote and quote not in targets:raise ValueError('Goal quote not in final target action')
    if rating['goal_complete']=='yes' and not quote.strip():raise ValueError('Goal success needs final motion evidence')
    expected={x['fact_id'] for x in facts};observed=set()
    for fact in rating['fact_reviews']:
        if set(fact)!= {'fact_id','status','quote'} or fact['fact_id'] not in expected or fact['fact_id'] in observed:
            raise ValueError('Protected fact inventory invalid')
        observed.add(fact['fact_id'])
        if fact['status'] not in ('retained','omitted','contradicted','unknown'):raise ValueError('Invalid fact status')
        if not isinstance(fact['quote'],str) or fact['quote'] and fact['quote'] not in targets:raise ValueError('Protected fact quote not in final target action')
        if fact['status'] in ('retained','contradicted') and not fact['quote'].strip():raise ValueError('Fact verdict needs exact final evidence')
    if observed!=expected:raise ValueError('Missing protected fact verdict')
    quotes=rating['boundary_conflict_quotes']
    if not isinstance(quotes,list) or any(not isinstance(q,str) or not q.strip() or q not in final for q in quotes):
        raise ValueError('Boundary conflict quote invented')
    if rating['no_new_boundary_conflict']=='no' and len(set(quotes))<2:raise ValueError('New conflict needs distinct source clauses')
    statuses=[x['status'] for x in rating['fact_reviews']]
    if any(x in ('omitted','contradicted') for x in statuses):rating['protected_facts_preserved']='no'
    elif 'unknown' in statuses:rating['protected_facts_preserved']='unknown'
    elif facts:rating['protected_facts_preserved']='yes'
    return rating
