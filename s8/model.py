import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

input = torch.rand((8, 3, 256, 256))

print("Model: ", model(input))
print("Script model: ", script_model(input))

assert torch.allclose(model(input)[:5], script_model(input)[:5])