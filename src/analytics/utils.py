import numpy as np
from ..schemas.metrics.pr_curve import PRCurvePoint
from scipy.stats import hmean


def get_best_f1_score_point(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray
) -> PRCurvePoint:
    f1_scores = hmean(np.vstack((precision, recall)), axis=0)
    max_f1_index = np.argsort(f1_scores)[-1]
    best_point = PRCurvePoint(
        prcn=precision[max_f1_index],
        rcl=recall[max_f1_index],
        thrd=thresholds[max_f1_index],
        f1=f1_scores[max_f1_index],
    )
    return best_point
