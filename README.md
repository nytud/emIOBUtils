
# emIOBUtils

An IOB format converter and corrector

_emIOBUtils_ is the Python rewrite of [CoreNLP's IOBUtils](https://github.com/stanfordnlp/CoreNLP/blob/45fcc4ccfd02259e0f379db5c392a03075d5b21e/src/edu/stanford/nlp/sequences/IOBUtils.java) which is written in JAVA.
It can take any (possibly ill-formed) [IOB span input](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) and convert/correct it according to the specified output style.

The program is useful to check whether the specified input contains valid spans or is ill-formed. Also, it can reduce or refine the possible labels for a specific purpose.

The supported formats are the following: iob[12], ioe[12], io, sbieo/iobes, noprefix

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
It will also recognize BILOU format (B=B, I=I, L=E, O=O, U=S).

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

## License

This program is licensed under the GPL 3.0 license.

## Acknowledgement

The authors gratefully acknowledge the efforts of CoreNLP developers to develop the algorithm and release their code under a free license.

We dedicate this library to all fellows whoever started to write such converters on their own.
