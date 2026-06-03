"""
Train MiniGPT on a small text corpus (Shakespeare by default).
Run: python train.py
"""

import torch
import requests
from model import MiniGPT

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 128    # context length (tokens per sample)
BATCH_SIZE  = 32
EMBED_DIM   = 256    # model width
N_HEADS     = 4      # attention heads
N_LAYERS    = 6      # transformer blocks
DROPOUT     = 0.1
LR          = 3e-4
STEPS       = 5000
EVAL_EVERY  = 200
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def get_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        print("Downloading Shakespeare dataset...")
        text = requests.get(url, timeout=10).text
    except Exception:
        print("Download failed — using built-in sample text.")
        text = (
            "To be, or not to be, that is the question: "
            "Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, "
            "Or to take arms against a sea of troubles "
            "And by opposing end them. To die—to sleep, "
            "No more; and by a sleep to say we end "
            "The heart-ache and the thousand natural shocks "
            "That flesh is heir to: 'tis a consummation "
            "Devoutly to be wish'd. To die, to sleep; "
            "To sleep, perchance to dream—ay, there's the rub, "
            "For in that sleep of death what dreams may come "
        ) * 200  # repeat to give the model enough text
    return text


def encode_decode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return encode, decode, len(chars)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, iters=50):
    model.eval()
    results = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item()
                  for _ in range(iters)]
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    print(f"Device: {DEVICE}")

    text = get_data()
    encode, decode, vocab_size = encode_decode(text)
    print(f"Corpus: {len(text):,} chars | vocab: {vocab_size} unique chars")

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_EVERY == 0 or step == 1:
            metrics = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            print(f"step {step:4d}/{STEPS}  train={metrics['train']:.4f}  val={metrics['val']:.4f}")

    # ── Generate sample text ──────────────────────────────────────────────────
    print("\n--- Generated text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start token
    generated_ids = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)
    print(decode(generated_ids[0].tolist()))

    torch.save(model.state_dict(), "minigpt.pt")
    print("\nModel saved to minigpt.pt")


if __name__ == "__main__":
    main()
