import torch
import time
import shutil

from typing import Callable

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Hyperparameters
    batch_size = 64
    block_size = 256
    max_iters = 0
    eval_interval = 500
    learning_rate = 3e-4
    dropout = 0.2
    eval_iters = 200
    number_layers = 6
    number_heads = 6
    number_embeddings = 384  # 384 / 6 = 64 dimensional heads
    # --------------

    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

    # The properties of the test
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)

    # Encoding and decoding
    string_to_int = {character: index for index, character in enumerate(chars)}
    int_to_string = {index: character for index, character in enumerate(chars)}
    encode = lambda string: [string_to_int[char] for char in string]
    decode = lambda list_int: "".join([int_to_string[integer] for integer in list_int])

    # Train and test splits
    data = torch.tensor(encode(corpus), dtype=torch.long, device=device)
    number_train = int(0.9 * len(data))
    train_data = data[:number_train]
    validation_data = data[:number_train]

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        number_embeddings=number_embeddings,
        block_size=block_size,
        number_heads=number_heads,
        number_layers=number_layers,
        dropout=dropout,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train
    model.train_model(
        max_iters,
        train_data,
        validation_data,
        optimizer,
        batch_size,
        eval_interval,
        eval_iters,
    )

    # Generate from the model
    print("\nAn example of text generated from the model.")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, 300)[0].tolist()))

    print("\nNow we generate text forever.\n")
    model.generate_forever(decode, 0.1)


# Data loading
def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else validation_data
    indices_start = torch.randint(len(data) - block_size - 1, (batch_size,))
    contexts = torch.stack([data[i : i + block_size] for i in indices_start])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in indices_start])
    return contexts, targets


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> dict:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            contexts, targets = get_batch(
                split, block_size, batch_size, train_data, validation_data
            )
            _, loss = model(contexts, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(torch.nn.Module):
    """One head of self-attention"""

    def __init__(
        self, head_size: int, block_size: int, number_embeddings: int, dropout: float
    ) -> None:
        super().__init__()
        self.key = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.query = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.value = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Attention scores
        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) ----> (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = torch.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # Weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(torch.nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(
        self,
        number_heads: int,
        head_size: int,
        block_size: int,
        number_embeddings: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                Head(head_size, block_size, number_embeddings, dropout)
                for _ in range(number_heads)
            ]
        )
        self.projection = torch.nn.Linear(number_embeddings, number_embeddings)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output


class FeedForward(torch.nn.Module):
    """A simple feed-forward layer."""

    def __init__(self, number_embeddings: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(number_embeddings, 4 * number_embeddings),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * number_embeddings, number_embeddings),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(torch.nn.Module):
    """Transformer block communication followed by computation."""

    def __init__(
        self, number_embeddings: int, number_heads: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        assert number_embeddings % number_heads == 0
        head_size = number_embeddings // number_heads
        self.self_attention = MultiHeadAttention(
            number_heads, head_size, block_size, number_embeddings, dropout
        )
        self.feed_forward = FeedForward(number_embeddings, dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(number_embeddings)
        self.layer_norm_2 = torch.nn.LayerNorm(number_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class BigramLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        number_embeddings: int,
        block_size: int,
        number_heads: int,
        number_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.block_size = block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, number_embeddings)
        self.position_embedding_table = torch.nn.Embedding(
            block_size, number_embeddings
        )
        self.blocks = torch.nn.Sequential(
            *[
                Block(number_embeddings, number_heads, block_size, dropout)
                for _ in range(number_layers)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(number_embeddings)
        self.language_model_head = torch.nn.Linear(number_embeddings, vocab_size)

    def forward(self, contexts: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = contexts.shape
        # index_x and targets are both (batch_size, block_size)
        token_embeddings = self.token_embedding_table(contexts)  # (B, T, C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.language_model_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape

        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, contexts: torch.Tensor, max_new_tokens: int):
        # contexts is of size (B, T), it's the array of indices of the current context.
        for _ in range(max_new_tokens):
            logits, _ = self(
                contexts[:, -self.block_size :]
            )  # Get the predictions. (B, T, C)
            # We need to trim the time to the last at most block_size tokens due to the positional encoder.
            logits = logits[
                :, -1, :
            ]  # We focus on the last element only, as it's what we want.
            # It becomes size (B, C)
            probs = torch.softmax(
                logits, dim=-1
            )  # We transform the logits to probabilities.
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # We sample the next token, (B,1)
            contexts = torch.cat((contexts, idx_next), dim=1)  # (B, T+1)
        return contexts

    def generate_forever(self, decoder: Callable[[list[int]], str], sleep_time: float):
        message = ""
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        try:
            while True:
                context = self.generate(context[:, -self.block_size :], 1)
                next_symbol = decoder([context[0][-1].tolist()])
                if next_symbol == "\n":
                    message = ""
                    print("")
                else:
                    width = shutil.get_terminal_size()[0]
                    if len(message) == width:
                        message = next_symbol
                        print("")
                    else:
                        message += next_symbol
                    print(f"{message}", end="\r")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            return

    def train_model(
        self,
        number_iterations: int,
        train_data: torch.Tensor,
        validation_data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        eval_interval: int,
        eval_iters: int,
    ):
        last_val_loss = float("inf")
        iter = 0
        try:
            while True:

                # We evaluate loss on train and val once in a while
                if iter % eval_interval == 0 or iter == number_iterations - 1:
                    losses = estimate_loss(
                        self,
                        eval_iters,
                        self.block_size,
                        batch_size,
                        train_data,
                        validation_data,
                    )
                    print(
                        f"At step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}."
                    )
                    if losses["val"] >= last_val_loss:
                        print("Early stop.")
                        return
                    last_val_loss = losses["val"]

                # Get a batch of data
                contexts, targets = get_batch(
                    "train", self.block_size, batch_size, train_data, validation_data
                )

                # Optimize
                _, loss = self(contexts, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                iter += 1
                if iter == number_iterations:
                    return

        except KeyboardInterrupt:  # Allows for keyboard interruption without exiting the script.
            print("Training manually interrupted.")


if __name__ == "__main__":
    main()
