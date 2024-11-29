import importlib

def test_dependencies():
    assert importlib.util.find_spec("transformers"), \
        "transformers is not installed"
    assert importlib.util.find_spec("rdkit"), \
        "rdkit is not installed"
    assert importlib.util.find_spec("torch"), \
        "pytorch is not installed"
    assert importlib.util.find_spec("sklearn"), \
        "scikit-learn is not installed"
    assert importlib.util.find_spec("scipy"), \
        "scipy is not installed"

def test_cuda():
    import torch
    assert torch.version.cuda, "pytorch cuda version is not installed"
