# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXBUILD           = sphinx-build
SOURCEDIR             = source
BUILDDIR              = build
TEMPLATEDIR           = "${SOURCEDIR}/_templates"
#DOCSDIR               = ../docs
#SCRIPTSDIR            = ../scripts
#TUTORIALDIR           = "${SOURCEDIR}/tutorial"
#QS_TUTORIALDIR        = "../../alpha-quickstart/tutorials"
#TUTORIALBUILD         = "${SCRIPTSDIR}/transform_notebook.py"

SPHINXOPTS            =
SPHINXAPIDOCOPTS      = -e -t ${TEMPLATEDIR} --no-toc -o $(SOURCEDIR) -f

# export SPHINX_APIDOC_OPTIONS = members,undoc-members,inherited-members,no-show-inheritance

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	-rm -r "$(BUILDDIR)"
#	-rm -r "$(SOURCEDIR)"/api
#	-rm "$(SOURCEDIR)"/gamma.*.rst
#	-rm "$(SOURCEDIR)"/*.automodsumm

apidoc:
	sphinx-apidoc ${SPHINXAPIDOCOPTS} ../src

build:
	# run the Sphinx build for html
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

	# create a nojekyll file for gh-pages:
	touch "$(BUILDDIR)"/html/.nojekyll

	# clean up potentially pre-existing files in /docs and move build:
	#if [ -d "${DOCSDIR}" ]; then \
	#    rm -rf "${DOCSDIR}"/*.html "${DOCSDIR}"/*.js "${DOCSDIR}"/*.inv "${DOCSDIR}"/_* \
	#       "${DOCSDIR}"/api "${DOCSDIR}"/tutorial; \
	#fi
    # move the last build into /docs
	#if [ -d "../docs" ]; then mv "$(BUILDDIR)"/html/* ../docs/; fi


html: apidoc build


#tutorial:
	# build the tutorial notebooks
#	@$(TUTORIALBUILD) "${TUTORIALDIR}" "${QS_TUTORIALDIR}"
#	mv "${QS_TUTORIALDIR}/quick_start.ipynb" "${QS_TUTORIALDIR}/1-quick_start.ipynb"
#	mv "${QS_TUTORIALDIR}/sklearndf.ipynb" "${QS_TUTORIALDIR}/2-sklearndf.ipynb"
#	mv "${QS_TUTORIALDIR}/model.ipynb" "${QS_TUTORIALDIR}/3-model.ipynb"
#	mv "${QS_TUTORIALDIR}/simulation.ipynb" "${QS_TUTORIALDIR}/4-simulation.ipynb"


# maybe not needed?
html-gh-pages:
	# target to generate github pages build on branch "gh-pages"
	cd .. && \
	git checkout gh-pages && \
	rm -rf * && \
	git checkout master sphinx data scripts && \
	git reset HEAD && cd sphinx && make html && \
	cd .. && \
	mv -fv "./sphinx/$(BUILDDIR)"/html/* ./  && \
	mkdir -p _static/notebooks && \
	python scripts/transform_notebook.py sphinx/source/tutorial/ _static/notebooks/ && \
	rm -rf sphinx data scripts && git add -A

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
