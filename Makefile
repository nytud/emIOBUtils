DIR := ${CURDIR}
all:
	@echo "See Makefile for possible targets!"

dist/*.whl dist/*.tar.gz:
	@echo "Building package..."
	python3 setup.py sdist bdist_wheel

build: dist/*.whl dist/*.tar.gz

install-user: build
	@echo "Installing package to user..."
	pip3 install dist/*.whl

test:
	@echo "Running tests..."
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOBES --output-style IOBES | diff - $(DIR)/tests/test_data.out.iobes && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-SBIEO --output-style SBIEO | diff - $(DIR)/tests/test_data.out.sbieo && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOBE1 --output-style IOBE1 | diff - $(DIR)/tests/test_data.out.iobe1 && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOB1 --output-style IOB1 | diff - $(DIR)/tests/test_data.out.iob1 && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOB2 --output-style IOB2 | diff - $(DIR)/tests/test_data.out.iob2 && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOE1 --output-style IOE1 | diff - $(DIR)/tests/test_data.out.ioe1 && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IOE2 --output-style IOE2 | diff - $(DIR)/tests/test_data.out.ioe2 && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-BIO-CORRECTED --output-style BIO | diff - $(DIR)/tests/test_data.out.bio_corrected && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-IO --output-style IO | diff - $(DIR)/tests/test_data.out.io && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-NOPREFIX --output-style NOPREFIX | diff - $(DIR)/tests/test_data.out.noprefix && cd ${CURDIR}
	cd /tmp && python3 -m emiobutils -i $(DIR)/tests/test_data.in --input-field-name NP-BIO --output-field-name NP-BILOU --output-style BILOU | diff - $(DIR)/tests/test_data.out.bilou && cd ${CURDIR}

install-user-test: install-user test
	@echo "The test was completed successfully!"

ci-test: install-user-test

uninstall:
	@echo "Uninstalling..."
	pip3 uninstall -y emiobutils

install-user-test-uninstall: install-user-test uninstall

clean:
	rm -rf dist/ build/ emiobutils.egg-info/

clean-build: clean build
