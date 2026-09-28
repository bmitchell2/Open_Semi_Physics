# Open_Semi_Physics

Verified, reusable semiconductor physics code for the Semiconductor Notes knowledge base.

The Python package lives in the `semiconductor_lib/` subfolder (its own `pyproject.toml`, README, tests, and examples).

## Install

```bash
pip install "git+https://github.com/bmitchell2/Open_Semi_Physics.git#subdirectory=semiconductor_lib"
```

## Run the tests

```bash
git clone https://github.com/bmitchell2/Open_Semi_Physics.git
cd Open_Semi_Physics/semiconductor_lib
pip install -e . pytest
pytest tests/ -v
```

See `semiconductor_lib/README.md` for the module layout and verification notes.
