# Minimal makefile for Sphinx dumentation
#

# You can set these variables from the command line.
SPHINXOPTS    = "-W"
# SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = RoboOrchardCore
SOURCEDIR     = .
BUILDDIR      = ../build/docs_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir -p $(BUILDDIR)
	@PYTORCH_JIT=0 \
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -r $(BUILDDIR)/*
	@rm -r build

.PHONY: help Makefile
