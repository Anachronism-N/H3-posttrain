"""Authorized small metadata claim API backed by one durable node-local ledger."""
import argparse
import hmac
import json
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from stage2_dynamic_pool_v1 import DynamicPool


def serve(config):
    cfg=json.loads(Path(config).read_bytes());assert cfg['bind'] in ('28.216.17.70','28.216.19.213','127.0.0.1') and 1024<=cfg['port']<65536
    token=Path(cfg['token_path']).read_text().strip();assert len(token)>=32 and cfg['allowed_owners']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path not in ('/claim','/inspect'):self.send_error(404);return
            auth=self.headers.get('Authorization','')
            if not hmac.compare_digest(auth,'Bearer '+token):self.send_error(403);return
            try:
                n=int(self.headers.get('Content-Length','0'));assert 0<n<=8192
                x=json.loads(self.rfile.read(n));assert x['owner'] in cfg['allowed_owners'] and set(x)<=set(('owner','request_id','size'))
                p=DynamicPool(cfg['database'])
                try:
                    if self.path=='/claim':v=p.claim(x['owner'],x['request_id'],x.get('size',8))
                    else:v=p.inspect_claim(x['owner'],x['request_id'])
                finally:p.close()
                raw=json.dumps(v,ensure_ascii=False,allow_nan=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            except (ValueError,KeyError,AssertionError) as e:self.send_error(409,'Claim contract rejected; do not automatically retry')
    ThreadingHTTPServer((cfg['bind'],cfg['port']),Handler).serve_forever()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();serve(a.config)
