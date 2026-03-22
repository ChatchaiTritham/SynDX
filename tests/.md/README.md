# SynDX Test Suite

Comprehensive test suite for the SynDX framework with 80%+ code coverage.

## Test Organization

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared fixtures and configuration
├── test_shap_reweighter.py              # Unit tests for SHAP reweighting
├── test_differential_privacy.py         # Unit tests for DP-SGD
├── test_xai_fidelity.py                 # Unit tests for XAI fidelity metrics
├── test_counterfactual_validator.py     # Unit tests for counterfactual generation
├── test_diagnostic_evaluator.py         # Unit tests for diagnostic evaluation
├── test_integration.py                  # Integration tests for component interactions
└── README.md                            # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_shap_reweighter.py
pytest tests/test_integration.py
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only privacy-related tests
pytest -m privacy

# Run only XAI fidelity tests
pytest -m fidelity

# Run only validation tests
pytest -m validation
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=syndx --cov-report=html --cov-report=term

# View HTML coverage report
# Open htmlcov/index.html in browser
```

### Run Tests Verbosely

```bash
# Verbose output with local variables
pytest -vv -l

# Show print statements
pytest -s
```

### Run Fast Tests Only (Skip Slow Tests)

```bash
pytest -m "not slow"
```

## Test Configuration

### pytest.ini

- **Test Discovery**: Automatically discovers test_*.py files
- **Markers**: Custom markers for test categorization
- **Coverage**: 80% minimum coverage threshold
- **Logging**: Console output enabled

### .coveragerc

- **Branch Coverage**: Enabled for more thorough testing
- **Exclusions**: Test files, demo code, and debug statements
- **Reports**: HTML, XML, and terminal output

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `test_config`: Global test configuration
- `mock_archetype_data`: Mock archetype patient data
- `mock_synthetic_data`: Mock synthetic patient data
- `mock_patient_data`: Single patient for counterfactual testing
- `mock_gradients`: Per-sample gradients for DP testing
- `mock_nmf_matrices`: NMF factor matrices for interaction testing
- `temp_output_dir`: Temporary directory for test outputs

## Test Markers

### Available Markers

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for component interactions
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.validation`: Validation tests for data quality
- `@pytest.mark.visualization`: Tests for plotting and figures
- `@pytest.mark.privacy`: Tests for differential privacy mechanisms
- `@pytest.mark.fidelity`: Tests for XAI fidelity metrics
- `@pytest.mark.counterfactual`: Tests for counterfactual generation

### Example Usage

```python
@pytest.mark.unit
@pytest.mark.privacy
class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""

    def test_gradient_clipping(self):
        # Test implementation
        pass
```

## Writing New Tests

### Unit Test Template

```python
import pytest
from syndx.module_name import ClassName

@pytest.mark.unit
class TestClassName:
    """Test ClassName functionality."""

    def test_initialization(self):
        """Test class initialization."""
        obj = ClassName()
        assert obj is not None

    def test_method_name(self, mock_data_fixture):
        """Test specific method."""
        obj = ClassName()
        result = obj.method(mock_data_fixture)
        assert result == expected_value
```

### Integration Test Template

```python
import pytest
from syndx.module1 import Class1
from syndx.module2 import Class2

@pytest.mark.integration
class TestClass1AndClass2:
    """Test Class1 and Class2 integration."""

    def test_workflow(self, mock_data):
        """Test complete workflow."""
        obj1 = Class1()
        obj1.fit(mock_data)

        obj2 = Class2()
        result = obj2.process(obj1.output)

        assert result is not None
```

## Continuous Integration

### GitHub Actions

Add to `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest --cov=syndx --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Coverage Goals

- **Overall**: ≥80% code coverage
- **Unit Tests**: ≥90% coverage for individual modules
- **Integration Tests**: All major workflows tested
- **Critical Paths**: 100% coverage for privacy and validation

## Common Issues

### Import Errors

If you encounter import errors:

```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Missing Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov numpy pandas scikit-learn xgboost shap scipy
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Run with parallel execution
pip install pytest-xdist
pytest -n auto
```

## Test Metrics

### Current Coverage

- `shap_reweighter.py`: Target ≥90%
- `differential_privacy.py`: Target ≥90%
- `xai_fidelity.py`: Target ≥85%
- `counterfactual_validator.py`: Target ≥80%
- `diagnostic_evaluator.py`: Target ≥85%

### Test Statistics

Run tests and check:

```bash
pytest --cov=syndx --cov-report=term-missing
```

Expected output:
```
Name                                        Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
syndx/phase2_synthesis/shap_reweighter.py     120      12    90%   45-47
syndx/phase2_synthesis/differential_privacy.py 150      15    90%
syndx/phase3_validation/xai_fidelity.py        180      25    86%
...
-------------------------------------------------------------------------
TOTAL                                        1000     80    92%
```

## Debugging Tests

### Run Specific Test

```bash
pytest tests/test_shap_reweighter.py::TestSHAPReweighterInit::test_default_initialization -vv
```

### Drop into Debugger

```bash
pytest --pdb
```

### Show Print Statements

```bash
pytest -s tests/test_diagnostic_evaluator.py
```

### Capture Warnings

```bash
pytest -W default
```

## Contributing

When adding new features:

1. Write unit tests for new classes/functions
2. Add integration tests for new workflows
3. Update fixtures in `conftest.py` if needed
4. Run full test suite before committing
5. Ensure coverage stays above 80%

## Contact

For questions about the test suite:
- Author: Chatchai Tritham
- Date: 2026-01-25
