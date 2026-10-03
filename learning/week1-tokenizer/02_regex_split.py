"""
STEP 2: Why doesn't GPT-4 just run raw BPE merging over the whole byte stream?

If you let BPE merge freely across an entire document, you get garbage merges
like "dog." + " The" becoming a single token because they happened to appear
together a lot -- merging across a word and the space+capital-letter that
starts the NEXT sentence. That pollutes the vocab with junk that doesn't
generalize.

The fix (used by GPT-2, GPT-3.5, GPT-4/cl100k_base): split text into chunks
FIRST using a regex, and NEVER let a BPE merge cross a chunk boundary.
This is "pre-tokenization." BPE then only merges *within* each chunk.

Below is the EXACT pattern cl100k_base (GPT-3.5/GPT-4's tokenizer) uses,
pulled directly from the installed tiktoken package -- not from memory.
"""

import regex  # note: stdlib `re` can't do \p{L}/\p{N} unicode classes or
              # possessive quantifiers -- that's why tiktoken needs the
              # third-party `regex` module, not `re`.

GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"          # 1. contractions: 's 'd 'm 't 'll 've 're
    r"|[^\r\n\p{L}\p{N}]?+\p{L}++"    # 2. a word: optional leading symbol/space + letters
    r"|\p{N}{1,3}+"                   # 3. digits, chunked in groups of at most 3
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*+"   # 4. punctuation/symbol runs (optional leading space)
    r"|\s++$"                         # 5. trailing whitespace at end of string
    r"|\s*[\r\n]"                     # 6. whitespace ending in a newline
    r"|\s+(?!\S)"                     # 7. whitespace run NOT followed by non-space
    r"|\s"                            # 8. fallback: single whitespace char
)

compiled = regex.compile(GPT4_SPLIT_PATTERN)

examples = [
    "Hello world!",
    "don't stop believin'",
    "The price is $1234567 today.",
    "def foo():\n    return 42",
    "wow!!!    ...really?",
]

for text in examples:
    chunks = compiled.findall(text)
    print(f"text : {text!r}")
    print(f"chunks: {chunks}")
    print()

print("Key things to notice:")
print("1. \"don't\" -> ['don', \"'t\"]           <- contraction split off (rule 1)")
print("2. \"1234567\" -> ['123','456','7']      <- numbers capped at 3 digits (rule 3)")
print("   why cap at 3? unlimited-length numbers would blow up the vocab with")
print("   one token per distinct large number ever seen in training -- capping")
print("   forces big numbers to be built from a handful of reusable digit-group")
print("   tokens instead.")
print("3. \" world\" keeps its LEADING space attached to the word, not trailing.")
print("   This is why GPT token counts look weird if you print tokens naively --")
print("   most word tokens actually start with an invisible space character.")
print("4. \"    return\" (4-space indent) -> whitespace becomes its own chunk,")
print("   separate from 'return' -- this is very relevant for code models.")
