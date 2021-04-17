import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def percentage_agreement(predict, target):
    length = float(predict.shape[0])
    match = 0.0

    for pred, y in zip(predict, target):
        if pred == y:
            match += 1.0
    return match/length


def diff_in_count(predict, target):
    length = float(predict.shape[0])
    error = 0.0

    for pred, y in zip(predict, target):
        error += y-pred
    return error/length


def abs_diff_in_count(predict, target):
    return abs(diff_in_count(predict, target))


def mse_loss(predict, target):
    cost_mse = nn.MSELoss()
    return cost_mse(predict, target)
