from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from operator import itemgetter
import torch


def average_precision_k(targets, predictions, k):
    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def evaluate(ground_truth, predict_results, ks):
    # precision and recall shape batch_size * k
    precision = torch.from_numpy(np.array([0.0] * len(ks)))
    recall = torch.from_numpy(np.array([0.0] * len(ks)))
    mrr = torch.from_numpy(np.array([0.0] * len(ks)))
    precision_u = [[], [], []]
    recall_u = [[], [], []]

    sorted_meta_prediction = torch.argsort(predict_results, dim=0, descending=True)
    sorted_meta_prediction = sorted_meta_prediction.squeeze()
    for i, k in enumerate(ks):
        pred = sorted_meta_prediction[:k].tolist()
        num_hit = len(set(pred).intersection(set(ground_truth)))
        precision[i] += float(num_hit) / len(pred)
        precision_u[i].append(float(num_hit) / len(pred))
        recall[i] += float(num_hit) / len(ground_truth)
        recall_u[i].append(float(num_hit) / len(ground_truth))
        mrr[i] += float(get_mrr(pred, ground_truth))

    k = 10
    n_pred = 10
    n_lab = len(ground_truth)
    n = min(max(n_pred, n_lab), k)
    arange = np.arange(n, dtype=np.float32)
    arange = arange[:n_pred]
    denom = np.log2(arange + 2.)
    gains = 1. / denom
    sorted_meta_prediction = sorted_meta_prediction.detach().clone().cpu().numpy()
    dcg_mask = np.in1d(sorted_meta_prediction[:n], ground_truth)
    dcg = gains[dcg_mask].sum()

    max_dcg = gains[arange < n_lab].sum()
    ndcg = dcg / max_dcg

    res = sorted_meta_prediction[:10]
    apk = average_precision_k(ground_truth, res, k=np.inf)
    return precision, recall, apk, precision_u, recall_u, ndcg, mrr


# MRR
def get_mrr(pre, truth):
    """
    MRR
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) TOP-K indics predicted by the model
    :return: MRR(Float), the mrr score
    """
    if len(truth) > len(pre):
        pre = np.concatenate((np.array(pre), np.zeros(len(truth) - len(pre))))
    else:
        truth = np.concatenate((np.array(truth), np.zeros(len(pre) - len(truth))))
        pre = np.array(pre)

    # ranks of the targets, if it appears in your indices
    hits = torch.tensor(truth == pre).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / len(truth)
    return mrr