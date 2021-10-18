import importlib

def test_dependencies():
    assert importlib.util.find_spec("transformers"), \
        "transformers is not installed"
    assert importlib.util.find_spec("rdkit"), \
        "rdkit is not installed"
    assert importlib.util.find_spec("torch"), \
        "pytorch is not installed"
    assert importlib.util.find_spec("torchtext"), \
        "torchtext is not installed"
    assert importlib.util.find_spec("sklearn"), \
        "scikit-learn is not installed"
    assert importlib.util.find_spec("scipy"), \
        "scipy is not installed"
    import torchtext
    torchtext_version = torchtext.__version__.split('.')
    assert int(torchtext_version[0])<1 and int(torchtext_version[1])<10, \
        "torchtext version >= 0.10.0 contains backward incompatible changes, \
        please install an older version"

def test_cuda():
    import torch
    assert torch.version.cuda, "pytorch cuda version is not installed"
