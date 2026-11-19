"""Read one original MS sample from the portable shared snapshot or master-local cache."""
import argparse,hashlib,json,re,sqlite3,tarfile
from pathlib import Path
DATA=Path(__file__).resolve().parent.parent/'original_MS_221887_shared_v1'

def run(a):
    if not re.fullmatch('[A-Za-z0-9_-]+',a.id):raise ValueError('Invalid source id')
    row=None
    if a.master_local:
        db=sqlite3.connect('file:/root/stage2_motion_revision_inputs_20261028_v2/catalog.db?mode=ro',uri=True);r=db.execute('SELECT payload FROM sources WHERE id=?',(a.id,)).fetchone();db.close()
        if r:
            v=json.loads(r[0]);row=dict(id=v['id'],split=v['split'],video_path=v['video_path'],original_metadata_archive=v['revision_input_archive']['path'],original_metadata_member=v['revision_input_member'],original_metadata_sha256=v['revision_input_sha256'])
    else:
        with (DATA/'source_manifest.jsonl').open() as f:
            for line in f:
                v=json.loads(line)
                if v['id']==a.id:row=v;break
    if row is None:raise ValueError('Source not in prepared original221887 snapshot')
    archive=Path(row['original_metadata_archive'])
    if not archive.exists():raise FileNotFoundError('Shared pack is still copying. Read immediately on master N2 with --master-local; do not run models again.')
    with tarfile.open(archive,'r:') as tar:
        member=tar.getmember(row['original_metadata_member']);assert member.isfile() and 0<member.size<=2*2**20;raw=tar.extractfile(member).read(2*2**20+1)
    assert hashlib.sha256(raw).hexdigest()==row['original_metadata_sha256'],'Source metadata changed';value=json.loads(raw)
    body=value if a.part=='all' else value['draft']['events'] if a.part=='events' else value['plan'] if a.part=='plan' else value['draft']
    print(json.dumps(dict(id=row['id'],split=row['split'],video_path=row['video_path'],part=a.part,original_data=body),ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--part',choices=('draft','events','plan','all'),default='draft');p.add_argument('--master-local',action='store_true');run(p.parse_args())
