#!/usr/bin/env pyhton3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

from xtsv import build_pipeline, parser_skeleton, jnius_config


def main():
    argparser = parser_skeleton(description='emIOBUtils - an IOB style converter and corrector')
    argparser.add_argument('--input-field-name', help='The name of the input field to convert or correct',
                           required=True, metavar='FIELD-NAME')
    argparser.add_argument('--output-field-name', help='The name of the output field (must be unique)', required=True,
                           metavar='FIELD-NAME')
    argparser.add_argument('--output-style', help='The name of the output span notation style', required=True,
                           choices={'iob1', 'iob2', 'bio', 'ioe1', 'ioe2', 'io', 'sbieo', 'iobes', 'noprefix', 'bilou',
                                    'IOB1', 'IOB2', 'BIO', 'IOE1', 'IOE2', 'IO', 'SBIEO', 'IOBES', 'NOPREFIX', 'BILOU'},
                           metavar='STYLE')
    opts = argparser.parse_args()  # TODO: Add multiple modes...

    jnius_config.classpath_show_warning = opts.verbose  # Suppress warning.
    conll_comments = opts.conllu_comments

    # Set input and output iterators...
    if opts.input_text is not None:
        input_data = opts.input_text
    else:
        input_data = opts.input_stream
    output_iterator = opts.output_stream

    # Set the tagger name as in the tools dictionary
    used_tools = ['iobconv']
    presets = []

    # Init and run the module as it were in xtsv

    # The relevant part of config.py
    em_iobutils = ('emiobutils', 'EmIOBUtils', 'IOB style converter and corrector (EmIOBUtils)', (),
                   {'out_style': opts.output_style, 'source_fields': {opts.input_field_name},
                    'target_fields': [opts.output_field_name]})
    tools = [(em_iobutils, ('iobconv', 'emiobutils'))]

    # Run the pipeline on input and write result to the output...
    output_iterator.writelines(build_pipeline(input_data, used_tools, tools, presets, conll_comments))

    # TODO this method is recommended when debugging the tool
    # Alternative: Run specific tool for input (still in emtsv format):
    # output_iterator.writelines(process(input_iterator, inited_tools[used_tools[0]]))

    # Alternative2: Run REST API debug server
    # app = pipeline_rest_api('TEST', inited_tools, presets,  False)
    # app.run()


if __name__ == '__main__':
    main()
