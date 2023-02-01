"""
Placeholder unit test file
"""

import types


def test_model_is_importable_in_tests():
    import titanic_dummy_model

    assert isinstance(titanic_dummy_model, types.ModuleType)
