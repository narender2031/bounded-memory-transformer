import torch

from bounded_memory_transformer import TinyTransformerLM, TransformerConfig


def tiny_config(*, dropout: float = 0.0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=17,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dropout=dropout,
    )


def test_forward_returns_logits_loss_and_attention_probabilities() -> None:
    torch.manual_seed(1)
    model = TinyTransformerLM(tiny_config())
    inputs = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))

    output = model(inputs, targets, return_attentions=True)

    assert output.logits.shape == (2, 6, 17)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.attentions is not None and len(output.attentions) == 2
    assert output.attentions[0].shape == (2, 4, 6, 6)

    future_probabilities = torch.triu(output.attentions[0], diagonal=1)
    assert torch.count_nonzero(future_probabilities) == 0


def test_future_token_cannot_change_earlier_logits() -> None:
    torch.manual_seed(2)
    model = TinyTransformerLM(tiny_config()).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed_future = original.clone()
    changed_future[0, -1] = 9

    original_logits = model(original).logits
    changed_logits = model(changed_future).logits

    torch.testing.assert_close(original_logits[:, :-1], changed_logits[:, :-1])


def test_optimizer_step_updates_model_parameters() -> None:
    torch.manual_seed(3)
    model = TinyTransformerLM(tiny_config(dropout=0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, 17, (2, 8))
    targets = torch.randint(0, 17, (2, 8))
    original_weight = model.token_embedding.weight.detach().clone()

    output = model(inputs, targets)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()

    assert not torch.equal(original_weight, model.token_embedding.weight)


def test_generation_preserves_prefix_and_adds_requested_tokens() -> None:
    torch.manual_seed(4)
    model = TinyTransformerLM(tiny_config())
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate(prefix, max_new_tokens=5, temperature=1.0, top_k=5)

    assert generated.shape == (1, 8)
    assert torch.equal(generated[:, : prefix.size(1)], prefix)
