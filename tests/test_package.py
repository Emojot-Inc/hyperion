from hyperion import __version__


def test_package_imports_with_core_dependencies_only():
    assert __version__ == "0.1.0"
