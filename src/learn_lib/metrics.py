import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """
    Calculates and returns the metrics.

    Args:
        y_true (torch.Tensor): true labels
        y_pred (torch.Tensor): predicted labels
    Returns:
        dict[str, float]: metrics (accuracy, precision, recall, f1)
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, average="macro"),
        "recall": recall_score(y_true_np, y_pred_np, average="macro"),
        "f1": f1_score(y_true_np, y_pred_np, average="macro"),
    }
