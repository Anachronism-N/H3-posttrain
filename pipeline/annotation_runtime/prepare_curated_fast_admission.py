import hashlib,json
from pathlib import Path
RT=Path(__file__).resolve().parent;EXP=RT.parent
summary_path=EXP/'curated222939_inputs_v1/manifest.summary.json';summary=json.loads(summary_path.read_text())
source=Path(summary['manifest']['path']);h=hashlib.sha256();rows=[];seen=set()
with source.open('rb') as f:
 for line in f:
  h.update(line)
  if not line.strip():continue
  row=json.loads(line);assert row['source_ordinal']==len(rows) and row['id'] not in seen;seen.add(row['id']);rows.append({'id':row['id'],'source_ordinal':row['source_ordinal'],'split':row['split'],'batch_id':row['batch_id']})
assert h.hexdigest()==summary['manifest']['sha256'] and len(rows)==222939
raw=b''.join((json.dumps(r,separators=(',',':'))+chr(10)).encode() for r in rows);path=RT/'curated222939_admission_ids_v1.jsonl'
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'count':len(rows),'original_full_manifest_SHA_verified':True,'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest()}))
