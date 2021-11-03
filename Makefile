.DEFAULT_GOAL := all

NAME=$(shell basename `pwd`)
INPUT_DIR=data
OUTPUT_DIR=processed

SAMPLES=$(shell find data/ -type d ! -name ${INPUT_DIR} -exec basename {} \;)
MCDs=$(shell find data/ -name "*.mcd" -type f)
MODEL_FILE=_models/pan_dataset.ilp
QUERY_STRING=20200122_PD_L1_100_percent_case


help:  ## Display help and quit
	@echo Makefile for the $(NAME) project.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

requirements:  ## Install Python requirements
	pip install \
		-r requirements.txt

transfer:  ## Transfer data from wcm.box.com to local environment
	imctransfer \
		--token -k -q $${QUERY_STRING}

inspect:  ## Inspect MCD files
	@echo "Running inspect step for samples: $(SAMPLES)"
	imc inspect \
		$(shell find $${INPUT_DIR} -name "*.mcd" -type f)

prepare:  ## Run first step of convertion of MCD to various files
	@echo "Running prepare step for samples: $(SAMPLES)"
	mkdir -p $${OUTPUT_DIR}
	imc prepare \
		-n ${SAMPLES} \
		--ilastik \
		--n-crops 0 \
		--ilastik-compartment nuclear \
		$(shell find $${INPUT_DIR} -name "*.mcd" -type f)

prob:  ## Generate probability array for ROIs
	lib/external/ilastik-1.3.3post2-Linux/run_ilastik.sh \
		--headless \
		--readonly \
		--export_source probabilities \
		--project $${MODEL_FILE} \
		$(shell find $${OUTPUT_DIR} -name "*_ilastik_s2.h5" -type f)

segment:  ## Segment probabilities into cell masks
	@echo "Running segment step for samples: $(SAMPLES)"

	find $${OUTPUT_DIR} \
		-name "*_ilastik_s2_Probabilities.tiff" \
		-exec rename "s/_ilastik_s2_Probabilities/_Probabilities/g" \
		{} \;

	imc segment \
		--from-probabilities \
		-m deepcell \
		-c cytoplasm \
		$(shell find $${OUTPUT_DIR} -name "*_full.tiff" -type f)

process: inspect prepare prob segment

all: install transfer process

analysis:
	@echo "Running analysis"
	python -u src/case.PM2078.analysis.py


sync:  ## Sync code to SCU server
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)

.PHONY : all requirements transfer inspect prepare segment process analysis sync
