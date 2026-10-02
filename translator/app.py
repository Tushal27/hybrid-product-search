"""
English -> French translator web page, powered by our LoRA fine-tuned Qwen.

Usage (after translator/train.py has produced lora_en_fr.pt):
    python translator/app.py
    then open http://127.0.0.1:5000
"""

import re
import time
from flask import Flask, jsonify, request, Response
from common import load_translator, translate

print("Loading the fine-tuned translator...")
tokenizer, model = load_translator()
print("Ready.")

app = Flask(__name__)
MAX_CHARS = 4000


def split_sentences(text):
    """Keep line breaks, and split each line into sentences -- the model was trained on single sentences."""
    pieces = []   # (is_blank_line, sentence)
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            pieces.append(None)
            continue
        pieces.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
        pieces.append("\n")
    return pieces


@app.post("/translate")
def translate_route():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(translation="", ms=0)
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is too long (max {MAX_CHARS} characters)."), 400
    start = time.perf_counter()
    pieces = split_sentences(text)
    sentences = [p for p in pieces if p not in (None, "\n")]
    french = iter(translate(tokenizer, model, sentences))
    out = []
    for p in pieces:
        if p is None:
            out.append("\n")
        elif p == "\n":
            out.append("\n")
        else:
            out.append(next(french) + " ")
    return jsonify(translation="".join(out).strip(), ms=round((time.perf_counter() - start) * 1000))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>English to French</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --ink:#1c1b19; --muted:#77736b; --line:#e3e0d8; --accent:#2b4acb; --accent-ink:#fff; }
  @media (prefers-color-scheme: dark) { :root { --bg:#131312; --card:#1d1d1b; --ink:#ecebe6; --muted:#9a968c; --line:#34332f; --accent:#7d93f5; --accent-ink:#10131f; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; padding:24px 16px; }
  main { max-width:960px; margin:0 auto; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 20px; font-size:.9rem; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:700px) { .grid { grid-template-columns:1fr; } }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; display:flex; flex-direction:column; }
  .label { font-size:.75rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
  textarea, .out { width:100%; min-height:220px; font:inherit; color:inherit; background:transparent; border:0; outline:0; resize:vertical; white-space:pre-wrap; }
  .out.empty { color:var(--muted); }
  .bar { display:flex; align-items:center; gap:12px; margin-top:12px; flex-wrap:wrap; }
  button { font:inherit; border:1px solid var(--line); background:transparent; color:var(--ink); border-radius:8px; padding:8px 14px; cursor:pointer; }
  button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .meta { color:var(--muted); font-size:.85rem; margin-left:auto; }
  .err { color:#c0392b; }
</style></head>
<body><main>
  <h1>English → French</h1>
  <p class="sub">A Qwen2.5-0.5B model fine-tuned with LoRA to do one job: translate. Runs entirely on this computer.</p>
  <div class="grid">
    <section class="panel">
      <div class="label">English</div>
      <textarea id="src" placeholder="Type or paste English text..." autofocus></textarea>
      <div class="bar"><button class="primary" id="go">Translate</button><span class="meta">Ctrl+Enter</span></div>
    </section>
    <section class="panel">
      <div class="label">Français</div>
      <div class="out empty" id="dst">The translation appears here.</div>
      <div class="bar"><button id="copy" disabled>Copy</button><span class="meta" id="meta"></span></div>
    </section>
  </div>
</main>
<script>
const src = document.getElementById('src'), dst = document.getElementById('dst');
const go = document.getElementById('go'), copy = document.getElementById('copy'), meta = document.getElementById('meta');
async function run() {
  const text = src.value.trim(); if (!text) return;
  go.disabled = true; go.textContent = 'Translating...'; meta.textContent = ''; meta.className = 'meta';
  try {
    const r = await fetch('/translate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Request failed');
    dst.textContent = j.translation; dst.classList.remove('empty'); copy.disabled = false;
    meta.textContent = (j.ms / 1000).toFixed(1) + ' s';
  } catch (e) { meta.textContent = e.message; meta.className = 'meta err'; }
  go.disabled = false; go.textContent = 'Translate';
}
go.onclick = run;
src.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run(); });
copy.onclick = async () => { try { await navigator.clipboard.writeText(dst.textContent); copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1200); } catch (e) {} };
</script></body></html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
