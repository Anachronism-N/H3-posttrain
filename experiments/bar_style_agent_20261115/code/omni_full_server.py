"""Full Qwen3-Omni model, text-only generation, explicit chat template."""
import argparse, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

LOCK = threading.Lock(); STATE = {}


def load(path, per_gpu_gib):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tp = os.path.join(path, 'chat_template.json')
    if getattr(tokenizer, 'chat_template', None) is None and os.path.isfile(tp):
        with open(tp) as f:
            tokenizer.chat_template = json.load(f)['chat_template']
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    memory = {i: '%dGiB' % per_gpu_gib for i in range(len(visible))}
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map='auto',
        max_memory=memory, low_cpu_mem_usage=True).eval()
    model.config.enable_audio_output = False
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.enable_audio_output = False
    torch.cuda.empty_cache()
    STATE.update(tokenizer=tokenizer, model=model, device=model.device)
    print('LOADED', flush=True)
    msgs = [dict(role='user', content='Reply with the single word OK.')]
    p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(p, return_tensors='pt').to(model.device)
    with torch.no_grad():
        try:
            out = model.generate(**inp, max_new_tokens=16, do_sample=False, return_audio=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        except TypeError:
            out = model.generate(**inp, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if isinstance(out, tuple): out = out[0]
    print('SELFTEST %r' % tokenizer.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)[:80], flush=True)


def complete(body):
    messages = list(body.get('messages') or [])
    schema = (body.get('response_format') or {}).get('json_schema', {}).get('schema') if isinstance(body.get('response_format'), dict) else None
    if schema is not None:
        instruction = ('Return ONLY one JSON object, no prose and no code fence, matching this JSON Schema: '
                       + json.dumps(schema, ensure_ascii=False, separators=(',', ':')))
        if messages and messages[0].get('role') == 'system':
            messages[0] = dict(messages[0], content=messages[0]['content'] + '\n' + instruction)
        else:
            messages.insert(0, dict(role='system', content=instruction))
    if messages and messages[0].get('role') == 'system':
        system_text = messages[0]['content']
        rest = messages[1:]
        if rest and rest[-1].get('role') == 'user':
            rest[-1] = dict(rest[-1], content=system_text + '\n\n' + rest[-1]['content'])
        else:
            rest.insert(0, dict(role='user', content=system_text))
        messages = rest
    tokenizer, model = STATE['tokenizer'], STATE['model']
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    max_new = int(body.get('max_tokens') or 1024)
    with torch.no_grad():
        try:
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, return_audio=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        except TypeError:
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if isinstance(out, tuple): out = out[0]
    new = out[0, inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return dict(id='cmpl-%d' % int(time.time() * 1000), object='chat.completion', created=int(time.time()),
                model=body.get('model', 'local'),
                choices=[dict(index=0, message=dict(role='assistant', content=text),
                              finish_reason='length' if len(new) >= max_new else 'stop')],
                usage=dict(prompt_tokens=int(inputs['input_ids'].shape[1]), completion_tokens=int(len(new)),
                           total_tokens=int(inputs['input_ids'].shape[1] + len(new))))


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.rstrip('/') in ('/health', '/v1/models'):
            return self._send(200, dict(object='list', data=[dict(id='qwen3-omni-full', object='model')]))
        self._send(404, dict(error='nf'))
    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        try: body = json.loads(self.rfile.read(n) or b'{}')
        except Exception as e: return self._send(400, dict(error=str(e)))
        try:
            with LOCK: r = complete(body)
        except Exception as e:
            return self._send(500, dict(error=dict(message=repr(e))))
        self._send(200, r)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--port', type=int, default=8017)
    ap.add_argument('--device', default='auto'); ap.add_argument('--per-gpu-gib', type=int, default=36)
    a = ap.parse_args(); load(a.model, a.per_gpu_gib)
    print('serving', a.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
