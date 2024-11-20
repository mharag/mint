from tqdm import tqdm
import random
import torch
from mint.logger import Logger, LoggerConfig
from dataclasses import dataclass, field
from mint.config import get_config

@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    use_cuda: bool = True
    warmup_steps: int = 10000
    grad_clip: float = 1.0
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)


trainer_config = get_config(TrainerConfig)


class Trainer:
    def __init__(self, model, dataset, learning_rate, batch_size, use_cuda, warmup_steps, grad_clip, logger_config):
        self.model = model
        self.grad_clip = grad_clip

        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        # set warmup steps to 10000
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / warmup_steps, 1.0)
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model.to(self.device)

        self.logger = Logger(**logger_config)

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
            # entropy of distribution
            self.logger.log_scalar("predictions_entropy", -torch.sum(predictions * torch.log(predictions)).item())
            self.logger.log_text("source", str(source_tokens[0]))
            self.logger.log_text("target", str(target_tokens[0]))
            self.logger.log_text("prediction", str(torch.argmax(predictions[0], dim=-1)))


            steps += 1
            progress_bar.desc = f"avg loss: {epoch_loss / steps:.4f}"
