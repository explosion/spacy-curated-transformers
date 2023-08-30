<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 💫 🤖 spaCy Curated Transformers

This package provides [spaCy](https://github.com/explosion/spaCy) components and
architectures to use a curated set of transformer models via
[`curated-transformers`](https://github.com/explosion/curated-transformers) in
spaCy.

[![PyPi](https://img.shields.io/pypi/v/spacy-curated-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-curated-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-curated-transformers/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-curated-transformers/releases)

## Features

- Use pretrained models based on one of the following architectures to
  power your spaCy pipeline:
  - ALBERT
  - BERT
  - CamemBERT
  - RoBERTa
  - XLM-RoBERTa
- All the nice features supported by [`spacy-transformers`](https://github.com/explosion/spacy-transformers)
  such as support for Hugging Face Hub, **multi-task learning**, the extensible config system and
  out-of-the-box serialization
- Deep integration into spaCy, which lays the groundwork for deployment-focused features
  such as distillation and quantization
- Minimal dependencies

## ⏳ Installation

Installing the package from pip will automatically install all dependencies.

```bash
pip install spacy-curated-transformers
```

## 🚀 Quickstart

An example project is provided in the [`project`](project) directory.

## 📖 Documentation

- 📘
  [Layers and Model Architectures](https://spacy.io/usage/layers-architectures):
  Power spaCy components with custom neural networks
- 📗 [`CuratedTransformer`](https://spacy.io/api/curatedtransformer): Pipeline component API
  reference
- 📗
  [Transformer architectures](https://spacy.io/api/architectures#curated-trf):
  Architectures and registered functions

## Bug reports and other issues

Please use [spaCy's issue tracker](https://github.com/explosion/spaCy/issues) to
report a bug, or open a new thread on the
[discussion board](https://github.com/explosion/spaCy/discussions) for any other
issue.
