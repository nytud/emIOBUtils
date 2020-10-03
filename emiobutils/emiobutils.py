#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
from itertools import tee, islice, chain


class EmIOBUtils:
    """
     * A class with functions to convert lists of labels between
       different IOB-style representations.
     * The original implementation is part of CoreNLP in JAVA
       created by Christopher Manning
       https://github.com/stanfordnlp/CoreNLP/blob/45fcc4ccfd02259e0f379db5c392a03075d5b21e/src/edu/stanford/nlp/sequences/IOBUtils.java

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
     * convenient. Note that the background label is never given a prefix.
     * This code is very specific to the particular CoNLL way of labeling
     * classes for IOB-style encoding, but this notation is quite widespread.
     * It will work on any of these styles of input.
     * This will also recognize BILOU and IOBE1 format (B=B, I=I, L=E, O=O, U=S=1).
     * It also works with lowercased names like i-org.
     * If the labels are not of the form 'C-Y+', where C is a single character,
     * then they will be regarded as NOPREFIX labels.
    """

    pass_header = True

    def __init__(self, out_style, strict_mode=True, unnest=False, background_label='O',
                 source_fields=None, target_fields=None):
        """
        This class is able to fix the ill-formed representation of the input
        and/or convert it according to the out_style parameter.
        It also can retrieve the list of chunks in order identified by their type, index and size.
        This class can be used with xtsv or standalone.

        :param out_style: one of iob[12], ioe[12], bio/iob, io, sbieo/iobes/iobe1, noprefix
        :param strict_mode: Raise ValueError on the first ill-formatted label or just warn to STDERR
        :param unnest: Does the input contain nested lists which is required to be unnested (e.g. sentences)
                       or just a stream of labels?
        :param background_label: The background label (e.g. O), which gets special treatment
        :param source_fields: xtsv related source fields
        :param target_fields: xtsv related target fields
        """
        self._style = {'iob1': self._iob1, 'iob2': self._iob2_bio, 'bio': self._iob2_bio, 'iob': self._iob2_bio,
                       'ioe1': self._ioe1, 'ioe2': self._ioe2, 'io': self._io, 'sbieo': self._sbieo_iobes,
                       'iobes': self._sbieo_iobes, 'iobe1': self._iobe1, 'noprefix': self._noprefix,
                       'bilou': self._bilou}

        # Input params
        self._conv_fun = self._style[out_style.lower()]
        if strict_mode:
            self._handle_invalid_tag_fun = self._handle_invalid_tag
        else:
            self._handle_invalid_tag_fun = self._handle_invalid_tag_dummy
        if unnest:
            self._unnest_fun = self._unnest
        else:
            self._unnest_fun = self._dummy_unnest
        self._background_label = background_label

        # Internal shortcuts
        self._bs1u = {'B', 'S', '1', 'U'}
        self._els1u = {'E', 'L', 'S', '1', 'U'}

        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = set()

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

    @classmethod
    def labels_to_entities(cls, seq, out_style, strict_mode=True, unnest=False, background_label='O'):
        """Alternative constructor to yield entities identified from the input sequence by their type, index and size
        :param seq: Input sequence
        :param out_style: one of iob[12], ioe[12], bio/iob, io, sbieo/iobes/iobe1, noprefix
        :param strict_mode: Raise ValueError on the first ill-formatted label or just warn to STDERR
        :param unnest: Does the input contain nested lists which is required to be unnested (e.g. sentences)
                       or just a stream of labels?
        :param background_label: The background label (e.g. O), which gets special treatment
        :return: generator of entities identified from the input sequence by their type, index and size
        """
        return cls(out_style, strict_mode, unnest, background_label).get_entities(seq)

    @classmethod
    def convert_format(cls, seq, out_style, strict_mode=True, unnest=False, background_label='O'):
        """Alternative constructor to yield converted/fixed labels from the input sequence with metadata
           (new_label, base, is_illformed, is_first, is_last)

        :param seq: Input sequence
        :param out_style: one of iob[12], ioe[12], bio/iob, io, sbieo/iobes/iobe1, noprefix
        :param strict_mode: Raise ValueError on the first ill-formatted label or just warn to STDERR
        :param unnest: Does the input contain nested lists which is required to be unnested (e.g. sentences)
                       or just a stream of labels?
        :param background_label: The background label (e.g. O), which gets special treatment
        :return: generator of converted labels and their metadata (new_label, base, is_illformed, is_first, is_last)
        """
        return cls(out_style, strict_mode, unnest, background_label).convert(seq)

    def process_sentence(self, sen, field_names):
        for tok, (out_label, *_) in zip(sen, self.convert([tok[field_names[0]] for tok in sen])):
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

    def convert(self, labels):
        """
        Convert between the representations and fix current one if needed (by converting it to valid format)
        :param labels: iterator on labels
        :return: the new label: str, the type of the current chunk: str,
                 is the label modified (for checking ill-formatted input when needed): bool
                 is the current label a starting label
                 is the curret label an ending label

        """
        bl_tuple = (self._background_label,)
        labels = self._unnest_fun(labels)
        labels1, labels2, labels3 = tee(labels, 3)               # Make a three-way tee from the iterator
        labels_prev = chain(bl_tuple, labels1)                   # Append background label to start
        labels_curr = labels2                                    # Do nothing
        labels_next = chain(islice(labels3, 1, None), bl_tuple)  # Skip first token and append background label to end

        # Iterate trhough [background_label, *labels, background_label] with a trigram window
        for previous_label, current_label, next_label in zip(labels_prev, labels_curr, labels_next):

            base, prefix = self._split_label(current_label)

            if base != self._background_label:
                previous_base, previous_prefix = self._split_label(previous_label)
                next_base, next_prefix = self._split_label(next_label)

                is_start_adjacent_same = previous_base == base and \
                    (prefix in self._bs1u or previous_prefix in self._els1u)
                is_end_adjacent_same = base == next_base and (next_prefix in self._bs1u or prefix in self._els1u)
                is_first = previous_base != base or is_start_adjacent_same
                is_last = base != next_base or is_end_adjacent_same

                new_label = self._conv_fun(is_first, is_last, is_start_adjacent_same, is_end_adjacent_same, base)
            else:
                new_label = base
                is_last = False  # Background label cannot be first or last
                is_first = False
            yield new_label, base, current_label != new_label, is_first, is_last

    def get_entities(self, seq):
        """Gets entities from sequence.

        Args:
            seq (list): sequence of labels.

        Returns:
            list: list of (chunk_type, chunk_start, chunk_size).

        Example:
            >>> from emiobutils import EmIOBUtils
            >>> e = EmIOBUtils(out_style='BIO')
            >>> seqv = ['B-PER', 'I-PER', 'O', 'B-LOC']
            >>> e.get_entities(seqv)
            [('PER', 0, 2), ('LOC', 3, 1)]
        """

        begin_offset = 0
        for i, (label, curr_type, is_ill_formatted, is_first, is_last) in enumerate(self.convert(seq)):
            if is_ill_formatted:
                self._handle_invalid_tag_fun(i)  # Validate!
            if is_last:
                yield curr_type, begin_offset, i + 1 - begin_offset  # Start and size difflib.SequenceMatcher
            if is_first:
                begin_offset = i

    @staticmethod
    def _unnest(seq):
        for sublist in seq:
            yield from sublist

    @staticmethod
    def _dummy_unnest(seq):
        return seq

    @staticmethod
    def _handle_invalid_tag(i):
        raise ValueError(f'The {i}th label is invalid!')

    @staticmethod
    def _handle_invalid_tag_dummy(i):
        print(f'The {i}th label is invalid!', file=sys.stderr)
