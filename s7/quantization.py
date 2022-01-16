import torch
import time

my_tensor = torch.rand((10))
print("Init tensor: ", my_tensor, my_tensor.dtype)

my_tensor_q = torch.quantize_per_tensor(my_tensor, 0.1, 10, torch.qint8)
print("Tensor after quantization: ", my_tensor_q, my_tensor_q.dtype)

my_tensor_dq = my_tensor_q.dequantize()
print("Tensor after dequantization: ", my_tensor_dq, my_tensor_dq.dtype)

print("--------------------------------------------------")

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()

# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights


input_fp32 = torch.randn(4, 4, 4, 4)

# run the models
start_int = time.time()

res_int = model_int8(input_fp32)

end_int = time.time()
print("Time quantized model: ", end_int - start_int)


start_float = time.time()

res_float = model_fp32(input_fp32)

end_float = time.time()
print("Time model: ", end_float - start_float)

