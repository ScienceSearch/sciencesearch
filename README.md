[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ScienceSearch/sciencesearch)

# ScienceSearch NLP Tools

**Note**: This repository is currently in _Alpha_ state. Things may not work!

Modules and scripts for ScienceSearch, including:

* Natural Language Processing (NLP) for metadata extraction
    - Run Yake, Rake, and KPMiner on texts
    - Simplify and automate hyperparameter training to select the 'best' algorithm

## Installation

### Developer install

First set up your preferred environment (e.g., [miniforge](https://github.com/conda-forge/miniforge)) with a supported version of Python.

Next install in editable mode with:

```shell
pip install -e .[dev]
```

Now you need to perform these additional manual install steps:

#### pke

From the [pke github page](https://github.com/boudinfl/pke), a command to install `spacy`:

```shell
python -m spacy download en_core_web_sm
```

### Rake stopwords

```shell
python -c "import nltk; nltk.download('stopwords')"
```

## Running

See the example notebooks in the `examples` directory.

You can run tests with `pytest` from the top-level directory.

## Documentation

Currently all documentation is in the source code.

> Read the source, Luke!

## Authors

Lawrence Berkeley National Laboratory:

- Oluwamayowa Amusat
- Anna Giannakou
- Dan Gunter
- Sufi Kaur
