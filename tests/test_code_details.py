# tests/test_code_details.py

import torch

from code_details import my_function_compiled, my_function_original, verify_results


def test_my_function_original():
    input_tensor = torch.randn(3, 4)
    result = my_function_original(input_tensor)
    assert result.shape == input_tensor.shape
    assert torch.allclose(result, input_tensor * input_tensor)


def test_my_function_compiled():
    input_tensor = torch.randn(3, 4)
    result = my_function_compiled(input_tensor)
    assert result.shape == input_tensor.shape
    assert torch.allclose(result, input_tensor * input_tensor)


def test_verify_results():
    input_tensor = torch.randn(3, 4)
    are_close, max_diff = verify_results(my_function_compiled, my_function_original, input_tensor)
    assert are_close
    assert max_diff < 1e-5
