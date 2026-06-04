# boolforge-uniform-samplers

Sampling canalizing and nested canalizing functions uniformly at random constitutes a non-trivial combinatorial problem. In a recent manuscript, we described methods to obtain an efficient uniform sampler, which is implemented in BoolForge. 

## Purpose

This repository contains all code to generate the figures presented in

> Ahana Ghosh and Claus Kadelka. *Uniform sampling of canalizing Boolean functions reveals hidden biases in Boolean network analysis*. Under review.


## Repository structure

```text
.
├── src/
├── figures/
├── requirements.txt
└── README.md
```

### scripts/

Source code used for all analyses.

### figures/

Generated manuscript figures.

## Installation

Clone the repository:

```bash
git clone https://github.com/username/boolforge-uniform-samplers.git
cd boolforge-uniform-samplers
```

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing results

To reproduce all manuscript figures:

```bash
python scripts/create_figures124_uniform_sampler_paper.py
python scripts/create_figure3_uniform_sampler_paper.py

```

Generated figures will be written to:

```text
figures/
```

## Dependencies

* Python 3.10
* NumPy
* Matplotlib
* BoolForge

## Authors

* Ahana Ghosh
* Claus Kadelka

## License

MIT License
