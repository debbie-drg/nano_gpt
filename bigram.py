import torch

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rage = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
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

# Data loading
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else validation_data
    indices_start = torch.randint(len(data) - block_size - 1, (batch_size,))
    contexts = torch.stack([data[i : i + block_size] for i in indices_start])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in indices_start])
    return contexts, targets


@torch.no_grad()
def estimate_loss(model: torch.nn.Module) -> dict:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            contexts, targets = get_batch(split)
            _, loss = model(contexts, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, contexts: torch.Tensor, targets: torch.Tensor | None = None):
        # index_x and targets are both (batch_size, block_size)
        logits = self.token_embedding_table(contexts)

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
            logits, _ = self(contexts)  # Get the predictions. (B, T, C)
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


model = BigramLanguageModel(vocab_size).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rage)

for iter in range(max_iters):

    # We evaluate loss on train and val once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"At step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}."
        )

    # Get a batch of data
    contexts, targets = get_batch("train")

    # Optimize
    _, loss = model(contexts, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
print("\nAn example of text generated from the model.")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 300)[0].tolist()))
