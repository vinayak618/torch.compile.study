import tempfile

import onnxruntime
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def onnxrt(gm, example_inputs, *, filename=None, provider=None):
    if filename is None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            return onnxrt(gm, example_inputs, filename=tmp.name)

    device_type = next(gm.parameters()).device.type
    input_names = [f"input_{i}" for i in range(len(example_inputs))]
    output_names = ["output"]

    # Export the model to ONNX
    torch.onnx.export(
        gm,
        example_inputs,
        filename,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch_size", 1: "sequence"} for name in input_names + output_names},
        opset_version=11,
    )

    if provider is None:
        provider = "CPUExecutionProvider" if device_type == "cpu" else "CUDAExecutionProvider"
    session = onnxruntime.InferenceSession(filename, providers=[provider])

    def _call(*args):
        ort_inputs = {name: arg.cpu().numpy() for name, arg in zip(input_names, args)}
        ort_outputs = session.run(None, ort_inputs)
        return tuple(torch.from_numpy(output).to(device_type) for output in ort_outputs)

    return _call


def generate_text(model, tokenizer, input_ids, max_length=50, temperature=1.0):
    for _ in range(max_length - len(input_ids[0])):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs[0][:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return input_ids


# Setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Compile the model
compiled_model = torch.compile(model, backend="onnxrt")

# Test generation
text = "The quick brown fox"
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate text using the compiled model
compiled_output = generate_text(compiled_model, tokenizer, input_ids)
compiled_generated_text = tokenizer.decode(compiled_output[0], skip_special_tokens=True)

print(f"Input text: {text}")
print(f"Compiled model generated text: {compiled_generated_text}")

# For comparison, generate text using the original PyTorch model
torch_output = generate_text(model, tokenizer, input_ids)
torch_generated_text = tokenizer.decode(torch_output[0], skip_special_tokens=True)

print(f"PyTorch model generated text: {torch_generated_text}")

# Compare outputs
print(f"\nOutputs match: {torch.all(compiled_output == torch_output)}")
if not torch.all(compiled_output == torch_output):
    print("Differences:")
    for i, (c, t) in enumerate(zip(compiled_output[0], torch_output[0])):
        if c != t:
            print(
                f"Position {i}: Compiled = {c.item()} ({tokenizer.decode(c)}), "
                f"PyTorch = {t.item()} ({tokenizer.decode(t)})"
            )
