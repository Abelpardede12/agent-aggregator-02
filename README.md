# agent-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)

A live free-model aggregator: one server speaks both the OpenAI and Anthropic
wires, and every completion it answers comes from a real free model on
OpenRouter, dialled through the grader's gateway. The model for each prompt is
picked from the brief's fourteen free models by house rating, filtered by what
the prompt needs (tools, JSON mode, vision), with failover to the next-ranked
model when the gateway refuses one.

## Run

```sh
python3 main.py --list   # print the registered catalog, exit 0
python3 main.py          # serve: register with $MAKEFUN_ENV_URL, poll /work, exit 0 when done
```

Python 3.13 stdlib only — no dependencies.

## How it works

- The brief (`$MAKEFUN_PROMPT_FILE`) is parsed at start: the model list is
  never hardcoded, so the catalog, the model ranking, and both `GET /v1/models`
  answers all come from the same fourteen ids.
- `POST /v1/chat/completions` and `POST /v1/messages` both dial
  `POST $OPENAI_BASE_URL/chat/completions` on the gateway; Anthropic requests
  are translated to Chat Completions and the answer translated back.
- Streaming is buffered and merged (content concatenated, tool calls merged by
  index, final usage kept), then re-rendered in the client's own wire, so the
  event order is always valid.
- A refused gateway call is retried on the next-ranked model (up to 3
  attempts); a served prompt carries the trail in
  `x-makefun-upstream-failures`, and giving up answers 502 in the client's
  error shape.
- Everything is deterministic: response ids are a sha256 of the canonical
  request plus the dialled model, `created` is 0, no wall clock or randomness
  in anything sent.
