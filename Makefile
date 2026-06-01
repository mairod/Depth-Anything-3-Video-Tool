PY      ?= python3
PIP     ?= $(VENV)/bin/pip
VENV    ?= .venv
DA3_URL := git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

.PHONY: install venv da3 clean

install: venv
	$(PIP) install -U pip wheel setuptools
	$(PIP) install -e .
	$(MAKE) da3
	$(MAKE) patch-skvideo

venv:
	@if [ ! -x "$(VENV)/bin/python" ]; then \
		echo "Creating $(VENV) with $(PY)"; \
		$(PY) -m venv $(VENV); \
	fi

# Install DA3 without its mac-hostile deps (xformers, open3d). The runtime deps
# DA3 actually uses are listed in pyproject.toml and were installed above.
da3:
	$(PIP) install --no-deps "depth-anything-3 @ $(DA3_URL)"

# sk-video 1.1.10 (RIFE dep) still references np.float / np.int aliases removed
# in numpy>=1.20. Patch the venv copy in place so RIFE runs on modern numpy.
patch-skvideo:
	@SKDIR="$$( $(VENV)/bin/python -c 'import skvideo, os; print(os.path.dirname(skvideo.__file__))' )"; \
	echo "Patching np.* aliases in $$SKDIR"; \
	find "$$SKDIR" -name '*.py' -exec perl -i -pe 's/\bnp\.(float|int|bool|complex|object|str|long|unicode)\b/$$1/g' {} +

clean:
	rm -rf $(VENV) build dist src/*.egg-info
