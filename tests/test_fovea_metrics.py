import numpy as np

from saccadenet.models.evaluate_fovea import classification_metrics


def test_classification_metrics_keep_blank_false_positives_separate():
    confusion = np.zeros((11, 11), dtype=int)
    for index in range(10):
        confusion[index, index] = 9
        confusion[index, 10] = 1
    confusion[10, 10] = 8
    confusion[10, 0] = 2
    result = classification_metrics(confusion)
    assert result["clear_digit_accuracy"] == 0.9
    assert result["blank_false_positive_rate"] == 0.2
    assert result["overall_accuracy"] == 98 / 110
    assert abs(result["macro_recall_11"] - (10 * 0.9 + 0.8) / 11) < 1e-12
