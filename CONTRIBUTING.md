# Want to Contribute? Awesome!

Thanks for considering contributing to SynDX! This started as a PhD project, but we'd love to have more eyes and hands on it.

## Getting Your Dev Environment Ready

Clone the repo and install in dev mode:

```bash
git clone https://github.com/ChatchaiTritham/SynDX.git
cd SynDX
pip install -e ".[dev]"
```

That `-e` flag is clutch—it means you can edit the code and see changes immediately without reinstalling.

## How We Write Code

We try to keep things readable:

- **PEP 8** is our friend (mostly—we're not obsessive about it)
- Run **Black** on your code before committing: `black syndx/`
- **Type hints** help, especially for complex functions
- Write **docstrings** for public-facing stuff so others know what's up

## Testing Your Changes

Make sure everything still works:

```bash
# Run all tests
pytest tests/

# Check test coverage
pytest --cov=syndx tests/
```

We're aiming for decent coverage, but don't stress if you can't test every edge case.

## Submitting Changes

Here's the flow:

1. **Fork** this repo to your GitHub account
2. **Create a branch** with a descriptive name (e.g., `fix-vae-loss-calc` or `add-mniere-archetype`)
3. **Make your changes** and write tests if you're adding features
4. **Run the linters and tests** to make sure nothing broke
5. **Open a PR** with a clear description of what you changed and why

We'll review it as soon as we can. Might ask for some tweaks—don't take it personally!

## Quick Note on Conduct

This is healthcare research code, so let's keep things professional and respectful. We're all here to build something useful for patients (eventually, once it's validated!).

---

Questions? Open an issue or email us. We don't bite.
