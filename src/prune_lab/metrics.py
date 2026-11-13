"""A selection score has an explicit identity; text overlap is not accuracy."""


def selection_score(metrics):
    key = "selection_score" if "selection_score" in metrics else "accuracy"
    return float(metrics[key])


def score_name(metrics):
    return metrics.get("score_name", "accuracy")
