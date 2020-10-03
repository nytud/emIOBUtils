
# emIOBUtils

A sequential labeling (IOB format) converter, corrector and evaluation package

_emIOBUtils_ is the Python rewrite of [CoreNLP's IOBUtils](https://github.com/stanfordnlp/CoreNLP/blob/45fcc4ccfd02259e0f379db5c392a03075d5b21e/src/edu/stanford/nlp/sequences/IOBUtils.java) which is written in JAVA.
It can take any (possibly ill-formed) [IOB span input](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) and convert/correct it according to the specified output style.

The program is useful to check whether the specified input contains valid spans or is ill-formed. Also, it can reduce or refine the possible labels for a specific purpose.

The supported formats are the following: iob[12], ioe[12], bio/iob, io, sbieo/iobes, noprefix.

The sequence evaluation metrics provided follows the naming convention of
 [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) and contains all metrics from
 [the current state of seqeval](https://github.com/chakki-works/seqeval/blob/ca1ad9f8d02a858af01f041aeb8efb7e00df6ac3/seqeval/metrics/sequence_labeling.py)
 with a few new metrics introduced. For more complex evaluation we recommend using [PyCM](https://github.com/sepandhaghighi/pycm) and [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)

### On IOB formats/styles

[The documentation of the original class](https://github.com/stanfordnlp/CoreNLP/blob/45fcc4ccfd02259e0f379db5c392a03075d5b21e/src/edu/stanford/nlp/sequences/IOBUtils.java#L30) presents the idea very smoothly:

A 4-way representation of all entities, like S-PERS, B-PERS,
I-PERS, E-PERS for a single word, beginning, internal, and end of an entity
(IOBES or SBIEO); always marking the first word of an entity (IOB2 or BIO);
only marking specially the beginning of non-first
items of an entity sequence with B-PERS (IOB1);
the reverse IOE1 and IOE2; IO where everything is I-tagged; and
NOPREFIX, where no prefixes are written on category labels.
The last two representations are deficient in not allowing adjacent
entities of the same class to be represented, but nevertheless
convenient. Note that the background label (e.g. O) is never given a prefix.
This code is very specific to the particular CoNLL way of labelling
classes for IOB-style encoding, but this notation is quite widespread.
It will work on any of these styles of input.
It will also recognize BILOU/IOBE1 format (B=B, I=I, L=E, O=O, U=S=1).

## Requirements

  - Python 3 (tested with 3.6)
  - Pip to install the additional requirements in requirements.txt

## Install on a local machine

  - Clone the repository: `git clone https://github.com/dlt-rilmta/emiobutils`
  - `sudo pip3 install dist/*.whl`
  - Use from Python 

## Usage

It is recommended to use the program as the part of [_e-magyar_ language processing framework](https://github.com/dlt-rilmta/emtsv).

If all input columns are already existing one can use `python3 -m emiobutils` with the unified [xtsv CLI API](https://github.com/dlt-rilmta/xtsv#command-line-interface).

### Mandatory CLI arguments

To use this library as a standalone tool the following CLI arguments must be supplied:

- `--input-field-name` to specify the name of the column to be processed in the input TSV file
- `--output-field-name` to specify the name of the column to put the input
- `--output-style` to specify the IOB format that the output must comply

### Available library functions

Conversion related:

- `EmIOBUtils`: The converter class
- `EmIOBUtils.convert_format()`: An alternative constructor for one-liner conversions
- `EmIOBUtils.labels_to_entities()`: An alternative constructor for one-liner entities generator from input label sequence

Evaluation related:

- `label_format_accuracy_score()`: This score counts the ratio of good vs. misfit labels identified by the EmIOBUtils converter
- `precision_recall_fscore_support()`: Compute precision, recall, f_beta-score and support for entities
- `f_score()`: Compute f_beta-score for entities
- `f1_score()`: Same as `f_score()` but beta is fixed to 1
- `precision_score()`: Compute precision for entities
- `recall_score()`: Compute recall entities
- `support()`: Compute support for entities
- `label_measures()`: Compute true positive, false positive, false negative, true negative, accuracy and specificity for labels
- `accuracy_score()`: Compute accuracy for labels
- `classification_error()`: Compute classification error for labels
- `specificity()`: Compute specificity for labels
- `performance_measure()`: Compute confusion matrix (true positive, false positive, false negative, true negative in dict format)
- `classification_report_vars()`: Compute the classification report metrics (precision, recall, f1-score, support for each label, micro, macro and weighted average) and return them as variables
- `classification_report()`: Like `classification_report_vars()` but returns formatted text report
- `test_metrics()`: Simple tests for the above metrics

## License

This program is licensed under the GPL 3.0 license.

## Acknowledgement

The authors gratefully acknowledge the efforts of CoreNLP developers to develop the algorithm and release their code under a free license.

We dedicate this library to all fellows whoever started to write such converters on their own.
