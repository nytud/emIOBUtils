#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

The code was originally taken from
https://github.com/chakki-works/seqeval/blob/ca1ad9f8d02a858af01f041aeb8efb7e00df6ac3/seqeval/metrics/sequence_labeling.py
then it was substantially overhauled.
"""


from collections import defaultdict, Counter

from emiobutils import EmIOBUtils


def label_format_accuracy_score(y_pred, labels_type, unnest=False, background_label='O'):
    """Compute the label format accuracy score

        The input labels must fit a label representation.
        This score counts the ratio of good vs. misfit labels identified by the EmIOBUtils converter.
        If this score is not 1, there is a serious problem with the tagger,
        because it could not maintain the correct representation of tags.

        Args:
            y_pred : 2d array. Estimated targets as returned by a tagger.
            labels_type: str. The type of the label representation to use.
            unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
            background_label: str. The label to used outside of entities.

        Returns:
            score : float.

        Example:
            >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> label_format_accuracy_score(pred_labels, labels_type='IOBES', unnest=True)
            0.8
    """
    e = EmIOBUtils(labels_type, unnest=unnest, background_label=background_label)
    c = Counter()
    for _, _, is_illformatted, _, _ in e.convert(y_pred):
        c[is_illformatted] += 1
    all_labels = c[True] + c[False]
    return c[False] / all_labels if all_labels > 0 else 1  # Good labels / all labels


def precision_recall_fscore_support(true_entities, pred_entities, beta):
    true_entities = set(true_entities)
    pred_entities = set(pred_entities)

    nb_correct = len(true_entities & pred_entities)  # TP
    nb_pred = len(pred_entities)  # TP + FP
    nb_true = len(true_entities)  # TP + FN

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f_beta_score = (1 + beta * beta) * p * r / ((beta * beta * p) + r) if p + r > 0 else 0

    return p, r, f_beta_score, nb_true


def f_score(y_true, y_pred, beta=1, labels_type='IOBES', strict_mode=True, unnest=False, background_label='O'):
    """Compute the F_beta score.

        The F_beta score can be interpreted as a weighted average of the precision and
        recall, where an F_beta score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F_beta score are
        equal. The formula for the F_beta score is:

            F1 = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)

        Args:
            y_true : 2d array. Ground truth (correct) target values.
            y_pred : 2d array. Estimated targets as returned by a tagger.
            beta: positive int. The beta weight factor.
            labels_type: str. The type of the label representation to use.
            strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
            unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
            background_label: str. The label to used outside of entities.

        Returns:
            score : float.

        Example:
            >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> f_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
            0.5
        """

    true_entities = EmIOBUtils.labels_to_entities(y_true, labels_type, strict_mode, unnest, background_label)
    pred_entities = EmIOBUtils.labels_to_entities(y_pred, labels_type, strict_mode, unnest, background_label)
    return precision_recall_fscore_support(true_entities, pred_entities, beta)[2]


def f1_score(y_true, y_pred, labels_type='IOBES', strict_mode=True, unnest=False, background_label='O'):
    return f_score(y_true, y_pred, strict_mode=strict_mode, labels_type=labels_type, unnest=unnest,
                   background_label=background_label)


def precision_score(y_true, y_pred, labels_type='IOBES', strict_mode=True, unnest=False, background_label='O'):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        labels_type: str. The type of the label representation to use.
        strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
        background_label: str. The label to used outside of entities.

    Returns:
        score : float.

    Example:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
        0.5
    """

    true_entities = EmIOBUtils.labels_to_entities(y_true, labels_type, strict_mode, unnest, background_label)
    pred_entities = EmIOBUtils.labels_to_entities(y_pred, labels_type, strict_mode, unnest, background_label)
    return precision_recall_fscore_support(true_entities, pred_entities, 1)[0]


def recall_score(y_true, y_pred, labels_type='IOBES', strict_mode=True, unnest=False, background_label='O'):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        labels_type: str. The type of the label representation to use.
        strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
        background_label: str. The label to used outside of entities.

    Returns:
        score : float.

    Example:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
        0.5
    """

    true_entities = EmIOBUtils.labels_to_entities(y_true, labels_type, strict_mode, unnest, background_label)
    pred_entities = EmIOBUtils.labels_to_entities(y_pred, labels_type, strict_mode, unnest, background_label)
    return precision_recall_fscore_support(true_entities, pred_entities, 1)[1]


def support(y_true, y_pred, labels_type='IOBES', strict_mode=True, unnest=False, background_label='O'):
    """Compute the number of occurrences of each class in y_true also known as support.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        labels_type: str. The type of the label representation to use.
        strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
        background_label: str. The label to used outside of entities.

    Returns:
        score : int.

    Example:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> support(true_labels, pred_labels, labels_type='IOB', unnest=True)
        0.5
    """

    true_entities = EmIOBUtils.labels_to_entities(y_true, labels_type, strict_mode, unnest, background_label)
    pred_entities = EmIOBUtils.labels_to_entities(y_pred, labels_type, strict_mode, unnest, background_label)
    return precision_recall_fscore_support(true_entities, pred_entities, 1)[3]


def label_measures(y_true, y_pred, unnest=False, background_label='O'):
    if unnest:
        y_true = (item for sublist in y_true for item in sublist)
        y_pred = (item for sublist in y_pred for item in sublist)

    tp, fp, fn, tn, nb_correct, count = 0, 0, 0, 0, 0, 0
    for y_t, y_p in zip(y_true, y_pred):
        tp += int(y_t == y_p and (y_t != background_label or y_p != background_label))
        fp += int(y_t != y_p and y_p != background_label)
        fn += int(y_t != background_label and y_p == background_label)
        tn += int(y_t == y_p == background_label)
        nb_correct += int(y_t == y_p)
        count += 1

    return tp, fp, fn, tn, nb_correct / count, tn / (tn + fp)


def accuracy_score(y_true, y_pred, unnest=False):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).

    Returns:
        score : float.

    Example:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(true_labels, pred_labels, unnest=True)
        0.8
    """

    return label_measures(y_true, y_pred, unnest)[4]


def classification_error(y_true, y_pred, unnest=False):
    """The ratio of predictions which were incorrect. Also known as misclassification rate"""

    return 1 - label_measures(y_true, y_pred, unnest)[4]  # 1 - accuracy_score


def specificity(y_true, y_pred, unnest=False):
    """The ratio of all negative samples are correctly predicted as negative. Also known as true negative rate (TNR)"""
    return label_measures(y_true, y_pred, unnest)[5]


def performance_measure(y_true, y_pred, unnest=False):
    """Compute the performance metrics: TP, FP, FN, TN

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).

    Returns:
        performance_dict : dict

    Example:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O', 'B-PER']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O', 'B-MISC']]
        >>> performance_measure(true_labels, pred_labels, unnest=True)
        {'TP': 3, 'FP': 3, 'FN': 1, 'TN': 4}
    """

    tp, fp, fn, tn = label_measures(y_true, y_pred, unnest)[:4]
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def classification_report_vars(y_true, y_pred, labels_type='IOBES', strict_mode=True, unnest=False,
                               background_label='O'):
    """Compute and return the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        labels_type: str. The type of the label representation to use.
        strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
        background_label: str. The label to used outside of entities.

    Returns:
        report : tuple. The name of each class, the precision, recall, F1 score for each class,
         the micro, macro and weighted average

    Examples:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> classification_report_vars(true_labels, pred_labels, labels_type='BIO', unnest=True)
        (['MISC', 'PER'], [0.0, 1.0], [0.0, 1.0], [0, 1.0], [1, 1], 0.5, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    """

    true_entities = set(EmIOBUtils.labels_to_entities(y_true, labels_type, strict_mode, unnest, background_label))
    pred_entities = set(EmIOBUtils.labels_to_entities(y_pred, labels_type, strict_mode, unnest, background_label))

    true_entities_per_type = defaultdict(set)
    for prev_type, begin_offset, end_offset in true_entities:
        true_entities_per_type[prev_type].add((begin_offset, end_offset))

    pred_entities_per_type = defaultdict(set)
    for prev_type, begin_offset, end_offset in pred_entities:
        pred_entities_per_type[prev_type].add((begin_offset, end_offset))

    true_type_names = sorted(true_entities_per_type.keys())

    ps, rs, f1s, s = [], [], [], []
    for type_name in true_type_names:
        p, r, f1, nb_true = precision_recall_fscore_support(true_entities_per_type[type_name],
                                                            pred_entities_per_type[type_name], beta=1)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    sum_p, sum_r, sum_f1 = precision_recall_fscore_support(true_entities, pred_entities, 1)[:3]
    sum_s = sum(s)
    avg_p = sum(ps) / len(s)
    avg_r = sum(rs) / len(s)
    avg_f1 = sum(f1s) / len(s)
    avgw_p = sum(i * j for i, j in zip(ps, s)) / sum_s
    avgw_r = sum(i * j for i, j in zip(rs, s)) / sum_s
    avgw_f1 = sum(i * j for i, j in zip(f1s, s)) / sum_s

    return true_type_names, ps, rs, f1s, s, sum_p, sum_r, sum_f1, sum_s, avg_p, avg_r, avg_f1, avgw_p, avgw_r, avgw_f1


def classification_report(y_true, y_pred, digits=2, labels_type='IOBES', strict_mode=True, unnest=False,
                          background_label='O'):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
        labels_type: str. The type of the label representation to use.
        strict_mode: bool. Raise ValueError on invalid label sequence or just print warning to stderr.
        unnest: bool. Are the input sequences nested (e.g. list of lists) or not (e.g. list of label sequences).
        background_label: str. The label to used outside of entities.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> classification_report(true_labels, pred_labels, labels_type='IOB', unnest=True)
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """

    type_names, ps, rs, f1s, s, sum_p, sum_r, sum_f1, sum_s, avg_p, avg_r, avg_f1, avgw_p, avgw_r, avgw_f1 \
        = classification_report_vars(y_true, y_pred, labels_type, strict_mode, unnest, background_label)

    name_width = max(len(prev_type) for prev_type in type_names)

    last_line_heading = 'weighted avg'
    width = max(name_width, len(last_line_heading), digits)
    report = '{:>{width}s}  {:>9} {:>9} {:>9} {:>9}\n\n'.\
        format('', 'precision', 'recall', 'f1-score', 'support', width=width)

    # Format scores for each type
    row_fmt = '{:>{width}s}  {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n'
    for type_name, p, r, f1, nb_true in zip(type_names, ps, rs, f1s, s):
        report += row_fmt.format(type_name, p, r, f1, nb_true, width=width, digits=digits)
    report += '\n'

    # compute averages
    report += row_fmt.format('micro avg', sum_p, sum_r, sum_f1, sum_s, width=width, digits=digits)
    report += row_fmt.format('macro avg', avg_p, avg_r, avg_f1, sum_s, width=width, digits=digits)
    report += row_fmt.format(last_line_heading, avgw_p, avgw_r, avgw_f1, sum_s, width=width, digits=digits)

    return report


def test_metrics():

    true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    print('true labels:', true_labels)
    print('pred labels:', pred_labels)
    lfas = label_format_accuracy_score(pred_labels, labels_type='IOBES', unnest=True)
    print('label_format_accuracy_score:', lfas)
    assert lfas == 0.8
    print('OK')
    fs = f_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
    print('f_score:', fs)
    assert fs == 0.5
    print('OK')
    ps = precision_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
    print('precision_score:', ps)
    assert ps == 0.5
    print('OK')
    rs = recall_score(true_labels, pred_labels, labels_type='IOB', unnest=True)
    print('recall_score:', rs)
    assert rs == 0.5
    print('OK')
    supp = support(true_labels, pred_labels, labels_type='IOB', unnest=True)
    print('support:', supp)
    assert supp == 2
    print('OK')
    accs = accuracy_score(true_labels, pred_labels, unnest=True)
    print('accuracy_score:', accs)
    assert accs == 0.8
    print('OK')
    ce = classification_error(true_labels, pred_labels, unnest=True)
    print('classification_error:', ce)
    assert ce == 0.19999999999999996
    print('OK')
    s = specificity(true_labels, pred_labels, unnest=True)
    print('specificity:', s)
    assert s == 0.6666666666666666
    print('OK')
    true_labels_pm = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O', 'B-PER']]
    pred_labels_pm = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O', 'B-MISC']]
    print('true labels:', true_labels_pm)
    print('pred labels:', pred_labels_pm)
    pm = performance_measure(true_labels_pm, pred_labels_pm, unnest=True)
    print('performance_measure:', pm)
    assert pm == {'TP': 3, 'FP': 3, 'FN': 1, 'TN': 4}
    print('OK')
    crv = classification_report_vars(true_labels, pred_labels, labels_type='BIO', unnest=True)
    print('classification_report_vars:', crv)
    assert crv == (['MISC', 'PER'], [0.0, 1.0], [0.0, 1.0], [0, 1.0], [1, 1],
                   0.5, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    print('OK')
    cr = classification_report(true_labels, pred_labels, labels_type='IOB', unnest=True)
    print('classification_report:', cr)
    assert cr == \
"""              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.50      0.50      0.50         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2
"""
    print('OK')
