import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        # x shape: [3, 4]
        y = self.linear(x)  # y shape: [3, 5]
        z = torch.sin(x)  # z shape: [3, 4]
        return y + z.matmul(self.linear.weight.t())  # Output shape: [3, 5]


model = SimpleModel()
x = torch.randn(3, 4)

# Compile the model with the onnxrt backend
compiled_model = torch.compile(model, backend="onnxrt")

# Run the compiled model
result = compiled_model(x)
