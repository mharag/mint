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
    max_steps_per_epoch: int | None = 0
    max_steps_per_validation: int | None = 0
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        learning_rate,
        batch_size,
        use_cuda,
        warmup_steps,
        grad_clip,
        max_steps_per_epoch,
        max_steps_per_validation,
        logger_config,
        source_tokenizer,
        target_tokenizer
    ):
        self.model = model
        self.grad_clip = grad_clip

        self.dataset = dataset
        self.train_data_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)
        self.test_data_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / warmup_steps, 1.0)
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.max_steps_per_epoch = max_steps_per_epoch
        self.max_steps_per_validation = max_steps_per_validation

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model.to(self.device)

        self.logger = Logger(**logger_config)

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            self.train_epoch(epoch, n_epochs)
            self.model.save(f"../checkpoint/epoch_{epoch}.pt")
            self.validate(epoch, n_epochs)

    def train_epoch(self, epoch, n_epochs):
        epoch_loss = 0
        steps = 0
        for batch in (progress_bar := tqdm(self.train_data_loader, total=self.max_steps_per_epoch)):
            source_tokens = torch.tensor(self.source_tokenizer.tokenize(batch["source"]), dtype=torch.long).to(self.device)
            target_tokens = torch.tensor(self.target_tokenizer.tokenize(batch["target"]), dtype=torch.long).to(self.device)
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

            if self.max_steps_per_epoch is not None and steps >= self.max_steps_per_epoch:
                break

            progress_bar.desc = f"Train {epoch+1}/{n_epochs} avg loss: {epoch_loss / steps:.4f}"

    @torch.no_grad()
    def validate(self, epoch, n_epoch):
        epoch_loss = 0
        steps = 0
        for batch in (progress_bar := tqdm(self.train_data_loader, total=self.max_steps_per_validation)):
            source_tokens = torch.stack(batch["source_tokens"], dim=1).to(self.device)
            target_tokens = torch.stack(batch["target_tokens"], dim=1).to(self.device)
            B, S = source_tokens.size()

            predictions = self.model(source_tokens, target_tokens)[:,
                          :-1]  # there is no ground truth for the last token
            loss = self.loss(predictions.reshape(B * S, -1), target_tokens.view(-1))

            epoch_loss += loss.item()
            self.logger.log_scalar("loss_val", loss.item())
            self.logger.log_text(
                "tokens_val",
                f"source: {str(source_tokens[0])} \n"
                f"target: {str(target_tokens[0])}"
                f"prediction: {str(torch.argmax(predictions[0], dim=-1))}"
            )
            source_text = self.source_tokenizer.detokenize(source_tokens[:1])[0]
            target_text = self.target_tokenizer.detokenize(target_tokens[:1])[0]
            prediction_text = self.target_tokenizer.detokenize(torch.argmax(predictions[0], dim=-1).unsqueeze(0))[0]
            self.logger.log_text(
                "translation_val",
                f"source: {source_text}\n"
                f"target: {target_text}\n"
                f" prediction: {prediction_text}"
            )

            self.logger.log_scalar("bleu_val", bleu(target_text, prediction_text))
            self.logger.log_scalar("chrf2_val", chrf2(target_text, prediction_text))

            steps += 1

            if self.max_steps_per_validation is not None and steps >= self.max_steps_per_validation:
                break

            progress_bar.desc = f"Validation {epoch}/{n_epoch} avg loss: {epoch_loss / steps:.4f}"

