import os
from typing import List

import onnxruntime as ort
import torch
import torch.fx as fx
from torch._dynamo import register_backend


def onnxrt_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(f"In onnxrt_compiler - Input type: {type(example_inputs[0])}")
    print(f"In onnxrt_compiler - Input shape: {example_inputs[0].shape}")

    # ONNX export
    torch.onnx.export(gm, example_inputs, "temp.onnx", opset_version=12)

    ort_session = ort.InferenceSession("temp.onnx")

    def run_onnx(*args):
        ort_inputs = {ort_session.get_inputs()[i].name: arg.numpy() for i, arg in enumerate(args)}
        ort_outputs = ort_session.run(None, ort_inputs)
        return torch.from_numpy(ort_outputs[0])  # Assuming single output

    return run_onnx


@register_backend
def onnxrt(gm, example_inputs):
    return onnxrt_compiler(gm, example_inputs)


@torch.compile(backend="onnxrt")
def my_function_compiled(x):
    print(f"In my_function - Input type: {type(x)}")
    print(
        f"In my_function - Input shape: {x.shape if isinstance(x, torch.Tensor) \
                                         else 'Not a tensor'}"
    )
    print(f"In my_function - Input content: {x}")
    if isinstance(x, list):
        print(f"List contents: {[type(item) for item in x]}")
    return x * x


def my_function_original(x):
    return x * x


def verify_results(compiled_func, original_func, input_tensor):
    with torch.no_grad():
        compiled_result = compiled_func(input_tensor)
        original_result = original_func(input_tensor)

    are_close = torch.allclose(compiled_result, original_result, rtol=1e-5, atol=1e-5)
    max_diff = torch.max(torch.abs(compiled_result - original_result))

    print(f"Results are close: {are_close}")
    print(f"Max difference: {max_diff}")

    return are_close, max_diff


# Test with both 1D and 2D tensors
for shape in [(10,), (3, 4), (2, 3, 4)]:
    print(f"\nTesting with shape {shape}")
    input_tensor = torch.randn(*shape)

    print("Original PyTorch function:")
    original_result = my_function_original(input_tensor)
    print(f"Result shape: {original_result.shape}")

    print("\nCompiled function:")
    try:
        compiled_result = my_function_compiled(input_tensor)
        print(f"Result shape: {compiled_result.shape}")

        print("\nVerification:")
        are_close, max_diff = verify_results(
            my_function_compiled, my_function_original, input_tensor
        )

        if not are_close:
            print("Warning: Results are not close enough!")
            print(f"Original result: {original_result}")
            print(f"Compiled result: {compiled_result}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()

# Clean up
if os.path.exists("temp.onnx"):
    os.remove("temp.onnx")
