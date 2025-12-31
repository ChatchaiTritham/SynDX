# Contributing to SynDX

We welcome contributions! Please follow these guidelines.

## Development Setup

```bash
git clone https://github.com/chatchai.tritham/SynDX.git
cd SynDX
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8
- Use Black for formatting: `black syndx/`
- Use type hints where appropriate
- Write docstrings for all public functions

## Testing

```bash
pytest tests/
pytest --cov=syndx tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run linting and tests
5. Submit PR with clear description

## Code of Conduct

Be respectful and professional. This is a research project for healthcare applications.
