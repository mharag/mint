from tqdm import tqdm
import random
import torch
from mint.logger import Logger, LoggerConfig
from dataclasses import dataclass, field
from mint.metrics import bleu, chrf2

@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    use_cuda: bool = True
    warmup_steps: int = 10000
    grad_clip: float = 1.0
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)


class Trainer:
    def __init__(self, model, dataset, learning_rate, batch_size, use_cuda, warmup_steps, grad_clip, logger_config, source_tokenizer=None, target_tokenizer=None):
        self.model = model
        self.grad_clip = grad_clip

        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / warmup_steps, 1.0)
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model.to(self.device)

        self.logger = Logger(**logger_config)

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            self.train_epoch()
            self.save_model(f"../checkpoint/epoch_{epoch}.pt")

    def train_epoch(self):
        epoch_loss = 0
        steps = 0
        for batch in (progress_bar := tqdm(self.data_loader)):
            source_tokens = torch.stack(batch["source_tokens"], dim=1).to(self.device)
            target_tokens = torch.stack(batch["target_tokens"], dim=1).to(self.device)
            B, S = source_tokens.size()

            predictions = self.model(source_tokens, target_tokens)[:,
                          :-1]  # there is no ground truth for the last token
            loss = self.loss(predictions.reshape(B*S, -1), target_tokens.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            self.logger.log_scalar("loss", loss.item())
            self.logger.log_scalar("lr", self.optimizer.param_groups[0]["lr"])
            self.logger.log_text(
                "tokens",
                f"source: {str(source_tokens[0])} \n"
                f"target: {str(target_tokens[0])}"
                f"prediction: {str(torch.argmax(predictions[0], dim=-1))}"
            )
            source_text = self.source_tokenizer.detokenize(source_tokens[:1])[0]
            target_text = self.target_tokenizer.detokenize(target_tokens[:1])[0]
            prediction_text = self.target_tokenizer.detokenize(torch.argmax(predictions[0], dim=-1).unsqueeze(0))[0]
            self.logger.log_text(
                "translation",
                f"source: {source_text}\n"
                f"target: {target_text}\n"
                f" prediction: {prediction_text}"
            )

            self.logger.log_scalar("bleu", bleu(target_text, prediction_text))
            self.logger.log_scalar("chrf2", chrf2(target_text, prediction_text))


            steps += 1
            progress_bar.desc = f"avg loss: {epoch_loss / steps:.4f}"
