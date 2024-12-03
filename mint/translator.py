import torch

class Translator:
    def __init__(self, model, source_tokenizer, target_tokenizer, search_strategy="greedy"):
        self.model = model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        if search_strategy == "greedy":
            self.search_strategy = self.greedy_search


    def translate(self, text, max_length=None):
        source_tokens = self.source_tokenizer.tokenize(text, max_length=max_length)
        source_tokens = torch.tensor(source_tokens).unsqueeze(0).to(next(self.model.parameters()).device)

        if max_length is None:
            max_length = self.model.config["max_seq_len"]

        print(source_tokens)
        encoder_output = self.model.encoder(source_tokens)

        translated_tokens = torch.empty(source_tokens.size()[0], 0, dtype=torch.long).to(source_tokens.device)
        for i in range(max_length):
            predictions = self.model.decoder(translated_tokens, encoder_output)
            translated_tokens = self.search_strategy(translated_tokens, predictions)
            if torch.all(translated_tokens[:, -1] == self.target_tokenizer.eos_token_id):
                break

        return self.target_tokenizer.detokenize(translated_tokens)


    def greedy_search(self, translated_tokens, predictions):
        return torch.cat([translated_tokens, torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(0)], dim=-1)

    def random_search(self, translated_tokens, predictions):
        return torch.cat([
            translated_tokens,
            torch.multinomial(torch.softmax(predictions[:, -1, :], dim=-1), 1)
        ], dim=-1)

    def beam_search(self, translated_tokens, predictions, n_beams=5):
        # batch dimension is beam dimension
        log_probs = torch.log_softmax(predictions, dim=-1)
        top_probs, top_tokens = torch.topk(log_probs[:, -1, :], n_beams, dim=-1)
        # for every every original beam, create n_beams new beams
        top_probs, top_tokens = top_probs.view(-1), top_tokens.view(-1)
        new_beans = torch.cat([
            torch.repeat_interleave(translated_tokens, n_beams, dim=0),
            top_tokens.unsqueeze(0)
        ], dim=-1)






