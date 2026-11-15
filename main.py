#!/usr/bin/env python3
"""Live free-model aggregator — make.fun free-model-aggregator-02.

One server, two wires (OpenAI + Anthropic). Every completion is dialled on the
grader's Chat Completions gateway from the fourteen free models the brief
lists, ranked by the house rating. Python 3.13 stdlib only.
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED_AT = "2026-11-15T00:00:00Z"
MAX_ATTEMPTS = 3
GATEWAY_TIMEOUT = 12
FAILOVER_BUDGET_S = 24.0
STOP_MAP = {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use", "content_filter": "refusal"}

STATE = {"models": [], "catalog": None}


def load_brief():
    """Parse the two single-line JSON docs out of MAKEFUN_PROMPT_FILE."""
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    models = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{") or '"schema"' not in line:
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                if doc.get("schema") == "free-model-aggregator-02/models/v1":
                    models = doc.get("models", [])
    return models


def build_catalog(models):
    return {"providers": [{"id": "openrouter",
                           "base_url": "https://openrouter.ai/api/v1",
                           "key_env": "OPENROUTER_API_KEY",
                           "wire": "openai",
                           "models": [{"id": m["id"], "free": True,
                                       "context_length": m.get("context_length")}
                                      for m in models]}]}


def rank_models(models, need_tools, need_json, need_vision):
    """Candidates meeting the requirements, rating desc, brief order tie-break."""
    cands = []
    for i, m in enumerate(models):
        if need_tools and not m.get("tools"):
            continue
        if need_json and not m.get("json"):
            continue
        if need_vision and not m.get("vision"):
            continue
        cands.append((-(m.get("rating_milli") or 0), i, m))
    cands.sort(key=lambda t: (t[0], t[1]))
    return [c[2] for c in cands]


def openai_requirements(body):
    need_tools = bool(body.get("tools"))
    need_json = bool(body.get("response_format"))
    need_vision = False
    for msg in body.get("messages", []) or []:
        c = msg.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    need_vision = True
    return need_tools, need_json, need_vision


def anthropic_to_openai(body):
    """Translate an Anthropic Messages body into a Chat Completions body."""
    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "".join(b.get("text", "") for b in system
                             if isinstance(b, dict) and b.get("type") == "text")
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []) or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        texts, images, tool_calls, tool_results = [], [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "image":
                src = b.get("source", {})
                if src.get("type") == "base64":
                    images.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (src.get("media_type", "image/png"),
                                                      src.get("data", ""))}})
                elif src.get("type") == "url":
                    images.append({"type": "image_url",
                                   "image_url": {"url": src.get("url", "")}})
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id"), "type": "function",
                                   "function": {"name": b.get("name"),
                                                "arguments": json.dumps(b.get("input", {}),
                                                                        sort_keys=True)}})
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                if not isinstance(c, str):
                    c = json.dumps(c)
                tool_results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                     "content": c})
        if role == "assistant" and tool_calls:
            messages.append({"role": "assistant", "content": "".join(texts),
                             "tool_calls": tool_calls})
        elif tool_results:
            messages.extend(tool_results)
        else:
            parts = []
            if texts:
                parts.append({"type": "text", "text": "".join(texts)})
            parts.extend(images)
            messages.append({"role": role,
                             "content": parts if images else "".join(texts)})
    out = {"messages": messages}
    tools = body.get("tools")
    if tools:
        out["tools"] = [{"type": "function",
                         "function": {"name": t["name"],
                                      "description": t.get("description", ""),
                                      "parameters": t.get("input_schema", {"type": "object"})}}
                        for t in tools]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            out["tool_choice"] = "auto"
        elif t == "any":
            out["tool_choice"] = "required"
        elif t == "none":
            out["tool_choice"] = "none"
        elif t == "tool":
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": tc.get("name")}}
    return out


def clamp_tokens(v, default=1024):
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = default
    return max(1, min(8192, n))


def det_id(prefix, body, model):
    canon = json.dumps({"body": body, "model": model},
                       sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(canon.encode()).hexdigest()[:24]


def norm_usage(usage):
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": max(0, pt), "completion_tokens": max(1, ct),
            "total_tokens": max(0, pt) + max(1, ct)}


def error_code(raw):
    try:
        doc = json.loads(raw.decode("utf-8", "replace"))
        err = doc.get("error", {})
        if isinstance(err, dict):
            return str(err.get("code") or err.get("type") or "http_error")
        return str(err)
    except Exception:
        return "http_error"


def gateway_call(model, oai_body, stream):
    url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/") + "/chat/completions"
    body = dict(oai_body)
    body["model"] = model
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
                                          "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=GATEWAY_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, json.dumps({"error": {"message": str(e),
                                        "code": "network_error"}}).encode()


def merge_stream(raw):
    """Buffer gateway SSE chunks and merge into one answer."""
    content_parts, tool_calls = [], {}
    finish_reason, usage = None, None
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        for ch in chunk.get("choices", []) or []:
            d = ch.get("delta", {}) or {}
            if d.get("content"):
                content_parts.append(d["content"])
            for tc in d.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                cur = tool_calls.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    cur["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    return "".join(content_parts), [tool_calls[i] for i in sorted(tool_calls)], \
        finish_reason, usage


def parse_answer(raw):
    doc = json.loads(raw.decode("utf-8", "replace"))
    ch = (doc.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    return (msg.get("content") or "", msg.get("tool_calls") or [],
            ch.get("finish_reason") or "stop", doc.get("usage") or {})


class AllFailed(Exception):
    def __init__(self, failures):
        self.failures = failures


def serve_completion(candidates, oai_body, stream):
    failures = []
    started = time.monotonic()
    for model in candidates[:MAX_ATTEMPTS]:
        if failures and time.monotonic() - started > FAILOVER_BUDGET_S:
            break
        status, raw = gateway_call(model["id"], oai_body, stream)
        if status == 200:
            if stream:
                content, tcs, fr, usage = merge_stream(raw)
            else:
                content, tcs, fr, usage = parse_answer(raw)
            return {"model": model["id"], "content": content, "tool_calls": tcs,
                    "finish_reason": fr or "stop", "usage": usage or {},
                    "failures": failures}
        failures.append({"status": status, "code": error_code(raw),
                         "model": model["id"]})
    raise AllFailed(failures)


def openai_response(result, body):
    msg = {"role": "assistant", "content": result["content"]}
    if result["tool_calls"]:
        msg["tool_calls"] = result["tool_calls"]
    fr = result["finish_reason"]
    if result["tool_calls"] and fr == "stop":
        fr = "tool_calls"
    return {"id": det_id("chatcmpl-", body, result["model"]),
            "object": "chat.completion", "created": 0, "model": result["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
            "usage": norm_usage(result["usage"])}


def openai_sse(result, body):
    rid = det_id("chatcmpl-", body, result["model"])
    fr = result["finish_reason"]
    if result["tool_calls"] and fr == "stop":
        fr = "tool_calls"
    delta = {"role": "assistant", "content": result["content"]}
    if result["tool_calls"]:
        delta["tool_calls"] = result["tool_calls"]
    out = ["data: " + json.dumps(
        {"id": rid, "object": "chat.completion.chunk", "created": 0,
         "model": result["model"],
         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"]
    out.append("data: " + json.dumps(
        {"id": rid, "object": "chat.completion.chunk", "created": 0,
         "model": result["model"],
         "choices": [{"index": 0, "delta": {}, "finish_reason": fr}],
         "usage": norm_usage(result["usage"])}) + "\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out)


def anthropic_response(result, body):
    content = []
    if result["content"]:
        content.append({"type": "text", "text": result["content"]})
    for tc in result["tool_calls"]:
        try:
            inp = json.loads(tc["function"]["arguments"] or "{}")
        except ValueError:
            inp = {}
        content.append({"type": "tool_use",
                        "id": tc.get("id") or det_id("toolu_", body, result["model"]),
                        "name": tc["function"]["name"], "input": inp})
    stop = "tool_use" if result["tool_calls"] else STOP_MAP.get(
        result["finish_reason"], "end_turn")
    u = norm_usage(result["usage"])
    return {"id": det_id("msg_", body, result["model"]), "type": "message",
            "role": "assistant", "model": result["model"], "content": content,
            "stop_reason": stop,
            "usage": {"input_tokens": u["prompt_tokens"],
                      "output_tokens": u["completion_tokens"]}}


def anthropic_sse(result, body):
    mid = det_id("msg_", body, result["model"])
    u = norm_usage(result["usage"])
    stop = "tool_use" if result["tool_calls"] else STOP_MAP.get(
        result["finish_reason"], "end_turn")
    ev = []
    ev.append("event: message_start\ndata: " + json.dumps(
        {"type": "message_start",
         "message": {"id": mid, "type": "message", "role": "assistant",
                     "model": result["model"], "content": [],
                     "stop_reason": None, "stop_sequence": None,
                     "usage": {"input_tokens": u["prompt_tokens"],
                               "output_tokens": 0}}}) + "\n\n")
    idx = 0
    if result["content"]:
        ev.append("event: content_block_start\ndata: " + json.dumps(
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}}) + "\n\n")
        ev.append("event: content_block_delta\ndata: " + json.dumps(
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": result["content"]}}) + "\n\n")
        ev.append("event: content_block_stop\ndata: " + json.dumps(
            {"type": "content_block_stop", "index": 0}) + "\n\n")
        idx = 1
    for tc in result["tool_calls"]:
        ev.append("event: content_block_start\ndata: " + json.dumps(
            {"type": "content_block_start", "index": idx,
             "content_block": {"type": "tool_use", "id": tc.get("id"),
                               "name": tc["function"]["name"], "input": {}}}) + "\n\n")
        ev.append("event: content_block_delta\ndata: " + json.dumps(
            {"type": "content_block_delta", "index": idx,
             "delta": {"type": "input_json_delta",
                       "partial_json": tc["function"]["arguments"]}}) + "\n\n")
        ev.append("event: content_block_stop\ndata: " + json.dumps(
            {"type": "content_block_stop", "index": idx}) + "\n\n")
        idx += 1
    ev.append("event: message_delta\ndata: " + json.dumps(
        {"type": "message_delta",
         "delta": {"stop_reason": stop, "stop_sequence": None},
         "usage": {"output_tokens": u["completion_tokens"]}}) + "\n\n")
    ev.append("event: message_stop\ndata: " + json.dumps(
        {"type": "message_stop"}) + "\n\n")
    return "".join(ev)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, raw, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, code, doc, extra=None):
        self._send(code, json.dumps(doc).encode(), "application/json", extra)

    @staticmethod
    def _failures_header(failures):
        if not failures:
            return {}
        return {"x-makefun-upstream-failures": json.dumps(failures)[:8192]}

    def do_GET(self):
        if self.path != "/v1/models":
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})
            return
        models = STATE["models"]
        if "x-api-key" in self.headers:
            data = [{"id": m["id"], "type": "model",
                     "display_name": m["id"], "created_at": FIXED_CREATED_AT}
                    for m in models]
            doc = {"data": data, "has_more": False, "object": "list",
                   "first_id": data[0]["id"] if data else None,
                   "last_id": data[-1]["id"] if data else None}
        else:
            doc = {"object": "list",
                   "data": [{"id": m["id"], "object": "model", "created": 0,
                             "owned_by": "openrouter"} for m in models]}
        self._send_json(200, doc)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            body = {}
        if self.path == "/v1/chat/completions":
            self._handle_openai(body)
        elif self.path == "/v1/messages":
            self._handle_anthropic(body)
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})

    def _candidates(self, need, preferred):
        cands = rank_models(STATE["models"], *need)
        if preferred and preferred != "auto":
            ids = [m["id"] for m in STATE["models"]]
            if preferred in ids:
                pref = STATE["models"][ids.index(preferred)]
                cands = [pref] + [m for m in cands if m["id"] != preferred]
        return cands

    def _handle_openai(self, body):
        stream = bool(body.get("stream"))
        oai = {"messages": body.get("messages", [])}
        oai["max_tokens"] = clamp_tokens(
            body.get("max_tokens", body.get("max_completion_tokens")))
        for k in ("temperature", "tools", "tool_choice", "response_format", "stop"):
            if body.get(k) is not None:
                oai[k] = body[k]
        cands = self._candidates(openai_requirements(body), body.get("model"))
        try:
            result = serve_completion(cands, oai, stream)
        except AllFailed as e:
            self._send_json(502, {"error": {"message": "all upstream models failed",
                                            "type": "server_error",
                                            "code": "upstream_failed"}},
                            self._failures_header(e.failures))
            return
        extra = self._failures_header(result["failures"])
        if stream:
            self._send(200, openai_sse(result, body).encode(),
                       "text/event-stream", extra)
        else:
            self._send_json(200, openai_response(result, body), extra)

    def _handle_anthropic(self, body):
        stream = bool(body.get("stream"))
        oai = anthropic_to_openai(body)
        oai["max_tokens"] = clamp_tokens(body.get("max_tokens"))
        if body.get("temperature") is not None:
            oai["temperature"] = body["temperature"]
        need_tools = bool(body.get("tools"))
        need_vision = False
        for msg in body.get("messages", []) or []:
            c = msg.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image":
                        need_vision = True
        cands = self._candidates((need_tools, False, need_vision), body.get("model"))
        try:
            result = serve_completion(cands, oai, stream)
        except AllFailed as e:
            self._send_json(502, {"type": "error",
                                  "error": {"type": "api_error",
                                            "message": "all upstream models failed"}},
                            self._failures_header(e.failures))
            return
        extra = self._failures_header(result["failures"])
        if stream:
            self._send(200, anthropic_sse(result, body).encode(),
                       "text/event-stream", extra)
        else:
            self._send_json(200, anthropic_response(result, body), extra)


def _post_json(url, doc, timeout=10):
    data = json.dumps(doc).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def register(env_url, port):
    for _ in range(5):
        try:
            _post_json(env_url + "/register",
                       {"protocol_version": 1, "port": port,
                        "catalog": STATE["catalog"]})
            return
        except Exception:
            time.sleep(0.5)


def work_done(env_url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(env_url + "/work", method="GET")
    with opener.open(req, timeout=10) as resp:
        return json.loads(resp.read().decode()).get("done") is True


def main():
    STATE["models"] = load_brief()
    STATE["catalog"] = build_catalog(STATE["models"])
    if "--list" in sys.argv:
        print(json.dumps(STATE["catalog"]))
        return 0
    port = int(os.environ.get("PORT", "0"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    bound = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if env_url:
        register(env_url, bound)
    while True:
        if env_url:
            try:
                if work_done(env_url):
                    break
            except Exception:
                pass
        time.sleep(0.4)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
