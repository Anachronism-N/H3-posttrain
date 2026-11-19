import copy,hashlib,json
from pathlib import Path
ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie")
MS=ROOT/'diff/logs_omnidance/exp/MS'
OUT=MS/'bar_annotation_optimized_20261024_v1'
FULL=MS/'chunk_caption12k_input_v1/manifest.jsonl'
SAME50=MS/'ms_truebar_first50_merge_short_edges_v1/windows.jsonl'
EXPECTED_FULL='4b2efc8233fed8dd7cdb46acf8da318db76cb575faa1177e9e2bc113d453d283'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=data:raise ValueError('Refusing changed input overwrite '+str(p))
  return
 with p.open('xb') as f:f.write(data)
def lines(rows):return ''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n' for r in rows).encode()
if sha(FULL)!=EXPECTED_FULL:raise ValueError('Full input changed')
rows=[json.loads(s) for s in FULL.read_text().splitlines() if s.strip()]
if len(rows)!=12000 or len({r['id'] for r in rows})!=12000:raise ValueError('Expected current12k source pool')
raw=SAME50.read_bytes();plans=[json.loads(s) for s in raw.decode().splitlines() if s.strip()]
if len(plans)!=50 or len({p['id'] for p in plans})!=50:raise ValueError('Same50 invalid')
source_by_id={r['id']:r for r in rows}
converted=[]
for plan in plans:
 sid=plan['id'];row=source_by_id[sid]
 if plan['binding']['id']!=sid:raise ValueError('Source mismatch')
 original=MS/'ms_truebar_first50_merge_short_edges_v1'/(sid+'.windows.json')
 receipt=MS/'ms_truebar_first50_merge_short_edges_v1'/(sid+'.receipt.json')
 if json.loads(original.read_text())!=plan:raise ValueError('Window original mismatch')
 converted.append(dict(schema='bar_visual_annotation_plan_v1',id=sid,manifest_record=row,
  binding=copy.deepcopy(plan['binding']),window_manifest=copy.deepcopy(plan),windows=copy.deepcopy(plan['windows']),
  plan_status=plan['status'],data_source='same50_detected_downbeat_merged',
  source_refs={'manifest':{'path':str(FULL),'sha256':EXPECTED_FULL},'windows':{'path':str(original),'sha256':sha(original)},
               'window_receipt':{'path':str(receipt),'sha256':sha(receipt)},'same50':{'path':str(SAME50),'sha256':hashlib.sha256(raw).hexdigest()}},
  semantic_approval=False,human_review_completed=False,formal_training=False))
save(OUT/'plan_first50.jsonl',lines(converted));save(OUT/'plan_first3.jsonl',lines(converted[:3]))
existing={p['id'] for p in plans}
pending=[r for r in rows if r['id'] not in existing]
for i in range(4):save(OUT/'runtime'/('beat_input_shard'+str(i)+'.jsonl'),lines(pending[i::4]))
summary=dict(schema='optimized_bar_inputs_v1',full_sources=len(rows),reuse_same50_sources=50,new_beat_sources=len(pending),
 beat_shard_counts=[len(pending[i::4]) for i in range(4)],first3_ids=[p['id'] for p in converted[:3]],
 plan_first3_sha256=sha(OUT/'plan_first3.jsonl'),plan_first50_sha256=sha(OUT/'plan_first50.jsonl'),
 original_fixed12k_or_labels_modified=False,semantic_approval=False,formal_training=False)
save(OUT/'input_summary.json',(json.dumps(summary,indent=2)+'\n').encode());print(json.dumps(summary))
