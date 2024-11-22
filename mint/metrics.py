import torch
from collections import Counter


def n_grams(sequence, n):
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def bleu(prediction, reference, max_n=4):
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(n_grams(prediction.split(), n))
        ref_ngrams = Counter(n_grams(reference.split(), n))

        overlap = sum(min(pred_ngrams[gram], ref_ngrams[gram]) for gram in pred_ngrams)
        total = sum(pred_ngrams.values())

        precisions.append(overlap / total if total > 0 else 0.0)

    log_precisions = torch.log(torch.tensor(precisions) + 1e-10) / max_n
    bleu = torch.exp(log_precisions.sum()).item()

    ref_length = len(reference)
    pred_length = len(prediction)
    if pred_length < ref_length:
        bleu *= torch.exp(torch.tensor(1 - ref_length / pred_length))

    return bleu


def chrf2(candidate, reference, n=6, beta=2):
    precision, recall = 0.0, 0.0

    for i in range(1, n + 1):
        candidate_ngrams = Counter(n_grams(candidate, i))
        reference_ngrams = Counter(n_grams(reference, i))

        overlap = sum((candidate_ngrams & reference_ngrams).values())
        total_candidate = sum(candidate_ngrams.values())
        total_reference = sum(reference_ngrams.values())

        precision = overlap / total_candidate if total_candidate > 0 else 0.0
        recall = overlap / total_reference if total_reference > 0 else 0.0

        precision += precision / n
        recall += recall / n

    if precision + recall > 0:
        f2_score = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    else:
        f2_score = 0.0

    return f2_score