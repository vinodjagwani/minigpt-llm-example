"""
Load trained MiniGPT and generate text.
Run: python generate.py
  or: python generate.py --prompt "To be or" --tokens 200 --temp 0.8
"""

import argparse
import torch
import requests
from model import MiniGPT

# must match values used during training
BLOCK_SIZE = 128
EMBED_DIM  = 256
N_HEADS    = 4
N_LAYERS   = 6
DROPOUT    = 0.0        # 0 at inference
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def get_vocab():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        text = requests.get(url, timeout=10).text
    except Exception:
        raise RuntimeError("Could not download text to rebuild vocab. Make sure you're online.")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",     type=str, default="\n",  help="Seed text")
    parser.add_argument("--tokens",     type=int, default=300,   help="Tokens to generate")
    parser.add_argument("--temp",       type=float, default=0.8, help="Temperature (0.1=focused, 1.5=wild)")
    parser.add_argument("--top_k",      type=int, default=40,    help="Top-k sampling")
    parser.add_argument("--checkpoint", type=str, default="minigpt.pt")
    args = parser.parse_args()

    stoi, itos, vocab_size = get_vocab()
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)

    model = MiniGPT(vocab_size, EMBED_DIM, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    print(f"Loaded {args.checkpoint}\n")

    prompt_ids = encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    out = model.generate(context, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k)

    print("--- Generated ---")
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
