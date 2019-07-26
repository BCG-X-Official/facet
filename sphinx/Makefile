# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# js: adapted to do apidoc, generate html and move into the docs folder
html: 
	# generate apidoc using docstrings
	sphinx-apidoc -e --no-toc -o "$(SOURCEDIR)" -f ./../src/gamma

	# run the Sphinx build for html
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# clean up potentially pre-existing files in /docs
	rm -rf ./../docs/*
	# move the last build into /docs
	mv "$(BUILDDIR)"/html/* ./../docs/
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
