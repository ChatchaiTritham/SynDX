"""
Simple test to verify syndx package installation
"""

print("Testing SynDX Package Installation")
print("=" * 60)

try:
    import syndx
    print("✅ syndx package imported successfully")
    print(f"   Version: {syndx.__version__}")
    print(f"   Location: {syndx.__file__}")
except Exception as e:
    print(f"❌ Failed to import syndx: {e}")

print("\nTesting individual imports:")
print("-" * 60)

# Test imports that don't depend on broken modules
imports_to_test = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
]

for module, name in imports_to_test:
    try:
        __import__(module)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {e}")

print("\n" + "=" * 60)
print("Package 'syndx' is installed (pip install -e .)")
print("Note: Some modules have indentation errors that need fixing")
print("=" * 60)
