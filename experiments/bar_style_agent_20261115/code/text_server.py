#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat-completions server for a local text LLM.

Text-only: messages are rendered with the tokenizer's chat template and the
model runs without images/audio. JSON-schema requests are not grammar-decoded;
the schema is appended to the system message as an explicit instruction, and
the caller still validates every field. Runs one request at a time.
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

CONDITIONAL = {"qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
               "qwen3_vl": "Qwen3VLForConditionalGeneration",
               "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration"}

LOCK = threading.Lock()
STATE = {}


def load(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kind = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if kind in CONDITIONAL:
        import os
        import transformers
        klass = getattr(transformers, CONDITIONAL[kind])
        extra = {}
        spec = os.environ.get("TEXT_SERVER_MAX_MEMORY")
        if spec:
            extra["max_memory"] = {int(part.split(":")[0]): part.split(":")[1] for part in spec.split(",")}
            extra["low_cpu_mem_usage"] = True
        model = klass.from_pretrained(model_path, torch_dtype=dtype, device_map=device,
                                      trust_remote_code=True, **extra)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device,
                                                     trust_remote_code=True)
    model.eval()
    import gc
    gc.collect(); torch.cuda.empty_cache()
    STATE.update(tokenizer=tokenizer, model=model, device=device)
    return model


class _JsonStop(StoppingCriteria):
    """Stop right after the first balanced JSON object is complete."""

    def __init__(self, tokenizer, prompt_length, closing_id):
        self.tokenizer, self.prompt_length, self.closing_id = tokenizer, prompt_length, closing_id
        self.seen_open = False

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1].item() != self.closing_id:
            return False
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        depth, in_string, escape = 0, False, False
        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
                self.seen_open = True
            elif char == "}":
                depth -= 1
                if self.seen_open and depth == 0:
                    return True
        return False


def first_json(text):
    """Return the first complete top-level JSON object; ignore trailing copies."""
    start = text.find("{")
    if start < 0:
        return text
    depth, in_string, escape = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text


def render(messages):
    return STATE["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def complete(body):
    messages = list(body.get("messages") or [])
    schema = None
    fmt = body.get("response_format") or {}
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")
    if schema is not None:
        # Not grammar-constrained: state the exact required shape in-band.
        instruction = ("Return ONLY one JSON object, no prose and no code fence, matching this JSON Schema: "
                       + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0], content=messages[0]["content"] + "\n" + instruction)
        else:
            messages.insert(0, dict(role="system", content=instruction))
    prompt = render(messages)
    tokenizer, model = STATE["tokenizer"], STATE["model"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature") or 0.0)
    seed = int(body.get("seed") or 0)
    kwargs = dict(max_new_tokens=max_new, do_sample=temperature > 0, num_beams=1,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if temperature > 0:
        kwargs.update(temperature=temperature)
    if schema is not None:
        closer = tokenizer.encode("}", add_special_tokens=False)
        if len(closer) == 1:
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_JsonStop(tokenizer, inputs["input_ids"].shape[1], closer[0])])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    new = output[0, inputs["input_ids"].shape[1]:]
    del output
    torch.cuda.empty_cache()
    text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    finish = "length" if len(new) >= max_new else "stop"
    if schema is not None:
        # Deployment boundary: keep only the first complete JSON object so a
        # self-repeating answer does not become an unparsable payload.
        text = first_json(text)
        if finish == "length":
            try:
                json.loads(text)
                # The object is complete; the extra tokens were trailing output.
                finish = "stop"
            except json.JSONDecodeError:
                pass
    return dict(id="cmpl-%d" % int(time.time() * 1000), object="chat.completion", created=int(time.time()),
                model=body.get("model", "local"),
                choices=[dict(index=0, message=dict(role="assistant", content=text), finish_reason=finish)],
                usage=dict(prompt_tokens=int(inputs["input_ids"].shape[1]), completion_tokens=int(len(new)),
                           total_tokens=int(inputs["input_ids"].shape[1] + len(new))))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._send(200, dict(object="list", data=[dict(id=STATE.get("name", "local"), object="model")]))
        if self.path.rstrip("/") in ("/health", "/"):
            return self._send(200, dict(status="ok"))
        self._send(404, dict(error="not found"))

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, dict(error="not found"))
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._send(400, dict(error=dict(message=str(error), type="invalid_request_error")))
        try:
            with LOCK:
                result = complete(body)
        except Exception as error:  # surfaced as a provider error, never retried here
            return self._send(500, dict(error=dict(message=repr(error), type="server_error")))
        self._send(200, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    load(args.model, args.device, dtype)
    STATE["name"] = args.model.rstrip("/").split("/")[-1]
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("serving", STATE["name"], "on", args.port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
