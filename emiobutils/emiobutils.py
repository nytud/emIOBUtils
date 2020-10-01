#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

"""
 * A class with functions to convert lists of labels between
   different IOB-style representations.
 * The original implementation is part of CoreNLP in JAVA
   created by Christopher Manning
   https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/sequences/IOBUtils.java
   at 45fcc4ccfd02259e0f379db5c392a03075d5b21e

 * This can be used to map from any IOB-style (i.e., 'I-PERS' style labels)
 * or just categories representation to any other.
 * It can read and change any representation to other representations:
 * a 4 way representation of all entities, like S-PERS, B-PERS,
 * I-PERS, E-PERS for single word, beginning, internal, and end of entity
 * (IOBES or SBIEO); always marking the first word of an entity (IOB2 or BIO);
 * only marking specially the beginning of non-first
 * items of an entity sequences with B-PERS (IOB1);
 * the reverse IOE1 and IOE2; IO where everything is I-tagged; and
 * NOPREFIX, where no prefixes are written on category labels.
 * The last two representations are deficient in not allowing adjacent
 * entities of the same class to be represented, but nevertheless
 * convenient.  Note that the background label is never given a prefix.
 * This code is very specific to the particular CoNLL way of labeling
 * classes for IOB-style encoding, but this notation is quite widespread.
 * It will work on any of these styles of input.
 * This will also recognize BILOU format (B=B, I=I, L=E, O=O, U=S).
 * It also works with lowercased names like i-org.
 * If the labels are not of the form 'C-Y+', where C is a single character,
 * then they will be regarded as NOPREFIX labels.
 *
 * @param labels List of labels in some style
 * @param style Output style; one of iob[12], ioe[12], io, sbieo/iobes, noprefix
 * @param background_label The background label (e.g. O), which gets special treatment
"""


class EmIOBUtils:
    pass_header = True

    def __init__(self, out_style, background_label='O', source_fields=None, target_fields=None):
        self._style = {'iob1': self._iob1, 'iob2': self._iob2_bio, 'bio': self._iob2_bio, 'ioe1': self._ioe1,
                       'ioe2': self._ioe2, 'io': self._io, 'sbieo': self._sbieo_iobes, 'iobes': self._sbieo_iobes,
                       'iobe1': self._iobe1, 'noprefix': self._noprefix, 'bilou': self._bilou}
        self._conv_fun = self._style[out_style.lower()]
        self._background_label = background_label

        self._bs1u = {'B', 'S', '1', 'U'}
        self._els1u = {'E', 'L', 'S', '1', 'U'}

        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = set()

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

    def process_sentence(self, sen, field_names):
        for tok, out_label in zip(sen, self._convert([tok[field_names[0]] for tok in sen])):
            tok.append(out_label)
        return sen

    def prepare_fields(self, field_names):
        return [field_names[next(iter(self.source_fields))]]  # Hack to handle any input fields

    @staticmethod
    def _iob1(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_start_adjacent_same:  # iob1, only B if adjacent
            prefix = 'B'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _iob2_bio(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_first:  # iob2 always B at start
            prefix = 'B'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _ioe1(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_end_adjacent_same:  # ioe1, only E if adjacent
            prefix = 'E'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _ioe2(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_last:  # ioe2 always E at end
            prefix = 'E'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _io(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _sbieo_iobes(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_first and is_last:
            prefix = 'S'
        elif not is_first and is_last:
            prefix = 'E'
        elif is_first and not is_last:
            prefix = 'B'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _iobe1(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_first and is_last:
            prefix = '1'
        elif not is_first and is_last:
            prefix = 'E'
        elif is_first and not is_last:
            prefix = 'B'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _noprefix(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        return base  # nothing to do as it's just base

    @staticmethod
    def _bilou(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base):
        if is_first and is_last:
            prefix = 'U'
        elif not is_first and is_last:
            prefix = 'L'
        elif is_first and not is_last:
            prefix = 'B'
        else:
            prefix = 'I'
        return f'{prefix}-{base}'

    @staticmethod
    def _split_label(label):
        if len(label) > 2 and label[1] == '-':
            base = label[2:]
            prefix = label[0].upper()
        else:
            base = label
            prefix = ' '
        return base, prefix

    def _convert(self, labels):
        background_label = self._background_label
        last_label_index = len(labels) - 1
        for i, current_label in enumerate(labels):
            if i > 0:
                previous_label = labels[i - 1]
            else:
                previous_label = background_label

            if i < last_label_index:
                next_label = labels[i + 1]
            else:
                next_label = background_label

            base, prefix = self._split_label(current_label)
            previous_base, previous_prefix = self._split_label(previous_label)
            next_base, next_prefix = self._split_label(next_label)

            is_start_adjacent_same = previous_base == base and (prefix in self._bs1u or previous_prefix in self._els1u)
            is_end_adjacent_same = base == next_base and (next_prefix in self._bs1u or prefix in self._els1u)
            is_first = previous_base != base or is_start_adjacent_same
            is_last = base != next_base or is_end_adjacent_same

            if base != background_label:
                new_label = self._conv_fun(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base)
            else:
                new_label = base
            yield new_label
