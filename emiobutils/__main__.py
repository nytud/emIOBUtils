#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

from xtsv import build_pipeline, parser_skeleton, jnius_config


def main():
    argparser = parser_skeleton(description='emIOBUtils - an IOB style converter and corrector')
    argparser.add_argument('--input-field-name', help='The name of the input field to convert or correct',
                           required=True, metavar='FIELD-NAME')
    argparser.add_argument('--output-field-name', help='The name of the output field (must be unique)', required=True,
                           metavar='FIELD-NAME')
    argparser.add_argument('--output-style', help='The name of the output span notation style', required=True,
                           choices={'iob1', 'iob2', 'bio', 'ioe1', 'ioe2', 'io', 'sbieo', 'iobes', 'iobe1', 'noprefix',
                                    'bilou', 'IOB1', 'IOB2', 'BIO', 'IOE1', 'IOE2', 'IO', 'SBIEO', 'IOBES', 'IOBE1',
                                    'NOPREFIX', 'BILOU'},
                           metavar='STYLE')
    opts = argparser.parse_args()  # TODO: Add multiple modes...

    jnius_config.classpath_show_warning = opts.verbose  # Suppress warning.

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
    output_iterator.writelines(build_pipeline(input_data, used_tools, tools, presets, opts.conllu_comments))

    # TODO this method is recommended when debugging the tool
    # Alternative: Run specific tool for input (still in emtsv format):
    # from xtsv import process
    # from emdummy import EmDummy
    # output_iterator.writelines(process(input_data, EmDummy(*em_dummy[3], **em_dummy[4])))

    # Alternative2: Run REST API debug server
    # from xtsv import pipeline_rest_api, singleton_store_factory
    # app = pipeline_rest_api('TEST', tools, {},  conll_comments=False, singleton_store=singleton_store_factory(),
    #                         form_title='TEST TITLE', doc_link='https://github.com/dlt-rilmta/emdummy')
    # app.run()


if __name__ == '__main__':
    main()
