# Repo template

## Description
- Code to analyze determinants of county-level measles outbreaks in the US from 2010 onwards
- A framework for outbreak size estimation based on latent $R_{eff}$ estimation from covariates

## How to run model code
1. This repo uses the [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager for Python. Please install `uv`.
2. A US Census API Key is also required. Keys can be requested [here](https://api.census.gov/data/key_signup.html). Please make a `.env` file in the repo root (akin to the `.env.example` file) and put your Census Key in the file.
2. Run `uv sync` from the project directory.
3. Run code! `uv run [FILE TO RUN]`. Alternatively, you can open a Python session with `uv run ipython` in your terminal to run code interactively.
