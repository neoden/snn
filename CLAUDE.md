# Claude Code Development Guidelines

This file contains conventions and guidelines for working on this project with Claude Code.

## Python Project Management

**Use `uv` for all Python project management tasks.**

- Add dependencies: `uv add <package>`
- Virtual environments: `uv venv`
- Running scripts: `uv run <script.py>`

Do not use `pip`, `conda`, or other package managers unless explicitly requested.

---

## Project Structure

This project contains multiple experimental implementations:

- `iris-nn/` - Backprop on Iris dataset (working)
- `iris-pg/` - REINFORCE on Iris dataset (working)
- `mnist-nn/` - Backprop on MNIST (working)
- `nmnist-stdp/` - R-STDP experiments (educational failure - see PROJECT_SUMMARY.md)

---

## Additional Guidelines

(To be expanded as project evolves)
