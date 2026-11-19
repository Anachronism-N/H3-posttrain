import hashlib,json
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');summary=json.loads((RT.parent/'curated222939_inputs_v1/manifest.summary.json').read_text());source=Path(summary['manifest']['path']);out=RT/'curated_source_index_v1.jsonl';rawsha=hashlib.sha256();rows=[];seen=set()
with source.open('rb') as f:
 for line in f:
  rawsha.update(line)
  if not line.strip():continue
  r=json.loads(line);assert r['source_ordinal']==len(rows) and r['id'] not in seen;seen.add(r['id']);rows.append({'id':r['id'],'source_ordinal':r['source_ordinal'],'batch_id':r['batch_id'],'split':r['split'],'raw_source_row_sha256':hashlib.sha256(line).hexdigest()})
assert len(rows)==222939 and rawsha.hexdigest()==summary['manifest']['sha256']
data=''.join(json.dumps(r,separators=(',',':'))+chr(10) for r in rows).encode()
if out.exists():assert out.read_bytes()==data
else:
 with out.open('xb') as f:f.write(data)
proof={'total':len(rows),'source_manifest':summary['manifest'],'index':{'path':str(out),'sha256':hashlib.sha256(data).hexdigest()},'each_index_ID_verified_from_raw_full_manifest':True,'full_original_manifest_hash_verified':True}
p=RT/'curated_source_index_v1.receipt.json';b=(json.dumps(proof,indent=2)+chr(10)).encode()
if p.exists():assert p.read_bytes()==b
else:
 with p.open('xb') as f:f.write(b)
print(json.dumps(proof))
