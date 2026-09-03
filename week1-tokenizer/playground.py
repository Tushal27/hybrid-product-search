"""
Interactive playground for OurGPT4Tokenizer. Run this, then edit `text`
below and re-run, or import the pieces into your own REPL.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os, importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("verify", os.path.join(_here, "05_verify_gpt4.py"))
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)  # this also runs the 8 built-in test cases

tok = verify.OurGPT4Tokenizer()

# ---- try your own text below ----
text = input("Enter the text:",)
ids = tok.encode(text)
print()
print(f"text  : {text!r}")
print(f"ids   : {ids}")
print(f"pieces: {[tok.decode([i]) for i in ids]}")
print(f"decode: {tok.decode(ids)!r}")

