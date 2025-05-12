# ScienceSearch NLP Tools

Modules and scripts for ScienceSearch, including:
* Natural Language Processing (NLP) for metadata extraction.

Code in this repository was originally written to generate metadata (keywords) for proposals from the Joint Genome Institute (JGI).

## Installation

### Developer install
First set up your preferred environment (e.g., [miniforge](https://github.com/conda-forge/miniforge)) with a supported version of Python.

Next install in editable mode with:
```
pip install -e .[dev]
```

Now you need to perform these additional manual install steps:

#### pke
From the [pke github page](https://github.com/boudinfl/pke), a command to install `spacy`:
```
python -m spacy download en_core_web_sm
```

### Rake stopwords
```
python -c "import nltk; nltk.download('stopwords')"
```

## Running

There is an example notebook under `examples/pipeline.ipynb` that runs using the data under
`data/jft`.

## Authors

Lawrence Berkeley National Laboratory:
- Oluwamayowa Amusat
- Anna Giannakou
- Dan Gunter
- Sufi Kaur

