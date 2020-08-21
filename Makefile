FACET_PATH := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../)
FACET_PATH := $(shell echo "${FACET_PATH}" | sed -e 's/ //g')
export CONDA_BLD_PATH=$(FACET_PATH)/facet/dist/conda

# absolute paths to local conda "channels" with built packages:
P_PYTOOLS=$(FACET_PATH)/pytools/dist/conda
P_SKLEARNDF=$(FACET_PATH)/sklearndf/dist/conda

# check local path for pytools packages and if they exist,
# add them as an conda channel:
ifneq ("$(wildcard $(P_PYTOOLS))","")
    C_PYTOOLS = -c "file:/$(P_PYTOOLS)"
endif

ifneq ("$(wildcard $(P_SKLEARNDF))","")
    C_SKLEARNDF = -c "file:/$(P_SKLEARNDF)"
endif

# the final command to append to conda build so that it finds locally
# built packages:
LOCAL_CHANNELS = $(C_PYTOOLS) $(C_SKLEARNDF)


help:
	@echo Usage: make package

.PHONY: help Makefile

clean:
	mkdir -p "$(CONDA_BLD_PATH)" && \
	rm -rf $(CONDA_BLD_PATH)/*

build:
	echo Creating a conda package for facet && \
	FACET_PATH="$(FACET_PATH)" conda-build -c conda-forge $(LOCAL_CHANNELS) conda-build/

package: clean build