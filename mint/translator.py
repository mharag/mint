import torch


class SearchStrategy:
    def __init__(self):
        pass

    def search(self, model, source_tokens, max_length=None, eos_token_id=None):
        if max_length is None:
            max_length = model.config["max_seq_len"]

        encoder_output = model.encoder(source_tokens)

        translated_tokens = torch.empty(source_tokens.shape[0], 0, dtype=torch.long).to(source_tokens.device)
        for i in range(max_length):
            if encoder_output.size(0) == 1:
                encoder_output = encoder_output.repeat(translated_tokens.size(0), 1, 1)

            predictions = model.decoder(translated_tokens, encoder_output)
            translated_tokens = self.step(translated_tokens, predictions)
            if torch.all(translated_tokens[:, -1] == eos_token_id):
                break

        return self.result(translated_tokens)

    def step(self, translated_tokens, predictions):
        raise NotImplementedError()

    def result(self, translated_tokens):
        return translated_tokens


class GreedySearch(SearchStrategy):
    def step(self, translated_tokens, predictions):
        return torch.cat([translated_tokens, torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(0)], dim=-1)

class RandomSearch(SearchStrategy):
    def step(self, translated_tokens, predictions):
        return torch.cat([
            translated_tokens,
            torch.multinomial(torch.softmax(predictions[:, -1, :], dim=-1), 1)
        ], dim=-1)


class BeamSearch(SearchStrategy):
    def __init__(self, n_beams):
        super(BeamSearch, self).__init__()
        self.n_beams = n_beams

    def step(self, translated_tokens, predictions):
        # batch dimension is beam dimension
        log_probs = torch.log_softmax(predictions, dim=-1)
        top_probs, top_tokens = torch.topk(log_probs[:, -1, :], self.n_beams, dim=-1)
        # for every every original beam, create n_beams new beams
        top_probs, top_tokens = top_probs.view(-1), top_tokens.view(-1)

        new_beans = torch.cat([
            torch.repeat_interleave(translated_tokens, self.n_beams, dim=0),
            top_tokens.unsqueeze(1)
        ], dim=-1)

        B, S = translated_tokens.size()
        S = S + 1  # new token

        batch_indexes = torch.repeat_interleave(torch.arange(B), self.n_beams, dim=-1)
        batch_indexes = torch.repeat_interleave(batch_indexes.unsqueeze(1), S, dim=-1)
        seq_indexes = torch.repeat_interleave(torch.arange(S).unsqueeze(0), B * self.n_beams, dim=0)

        scores = log_probs[batch_indexes, seq_indexes, new_beans]
        scores = torch.sum(scores, dim=-1)

        top_scores, top_indexes = torch.topk(scores, self.n_beams, dim=-1)

        return new_beans[top_indexes]

    def result(self, translated_tokens):
        return translated_tokens[:1]


class Translator:
    def __init__(self, model, source_tokenizer, target_tokenizer, search_strategy=None):
        self.model = model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        if search_strategy is None:
            self.search_strategy = GreedySearch()
        else:
            self.search_strategy = search_strategy


    def translate(self, text, max_length=None):
        source_tokens = self.source_tokenizer.tokenize(text, max_length=max_length)
        source_tokens = torch.tensor(source_tokens).unsqueeze(0).to(next(self.model.parameters()).device)

        translated_tokens = self.search_strategy.search(self.model, source_tokens, max_length=max_length, eos_token_id=self.target_tokenizer.eos_token_id)

        return self.target_tokenizer.detokenize(translated_tokens)
