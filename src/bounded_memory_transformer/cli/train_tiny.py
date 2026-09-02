"""Train the tiny Transformer on a character-level text file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from bounded_memory_transformer.tiny_transformer import (
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from bounded_memory_transformer.tiny_transformer.data import SequenceBatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        type=Path,
        default=Path("projects/01-tiny-transformer/corpus.txt"),
        help="UTF-8 training corpus",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--repeat-corpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--generate-tokens", type=int, default=240)
    parser.add_argument("--prompt", default="Memory")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLM,
    batcher: SequenceBatcher,
    *,
    batches: int,
) -> float:
    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = batcher.next_batch()
        output = model(inputs, targets)
        assert output.loss is not None
        losses.append(output.loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.repeat_corpus <= 0:
        raise ValueError("repeat-corpus must be positive")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    source_text = args.text.read_text(encoding="utf-8")
    text = source_text * args.repeat_corpus
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    split_at = int(0.9 * token_ids.numel())
    train_ids = token_ids[:split_at]
    validation_ids = token_ids[split_at:]
    if validation_ids.numel() <= args.context_length:
        raise ValueError("validation split is too short; increase --repeat-corpus")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    )
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_batcher = SequenceBatcher(
        train_ids,
        context_length=config.context_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    validation_batcher = SequenceBatcher(
        validation_ids,
        context_length=config.context_length,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        device=device,
    )

    print(
        f"device={device.type} parameters={model.parameter_count():,} "
        f"vocabulary={tokenizer.vocab_size} training_tokens={train_ids.numel():,}"
    )

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = train_batcher.next_batch()
        output = model(inputs, targets)
        assert output.loss is not None

        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            validation_loss = estimate_loss(
                model,
                validation_batcher,
                batches=args.eval_batches,
            )
            print(
                f"step={step:04d} train_loss={output.loss.item():.4f} "
                f"validation_loss={validation_loss:.4f}"
            )

    prompt = tokenizer.encode(args.prompt, device=device).unsqueeze(0)
    generated = model.generate(
        prompt,
        max_new_tokens=args.generate_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n--- sample ---")
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    main()
