import pickle
import numpy as np
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from train import EchoStateNetwork
sys.modules['__main__'].EchoStateNetwork = EchoStateNetwork

model_path = os.path.join(base_dir, "..", "software", "visualiser", "model_rc.pkl")
header_path = os.path.join(base_dir, "esn_estimator_weights.h")

print(f"Loading weights from {model_path}...")
with open(model_path, "rb") as f:
    package = pickle.load(f)

esn_soc = package['esn_soc']
esn_soh = package['esn_soh']
input_means = package['input_means']
input_stds = package['input_stds']

print("SOC Reservoir dimensions:")
print(f"  n_inputs: {esn_soc.n_inputs}")
print(f"  n_reservoir: {esn_soc.n_reservoir}")
print(f"  W_in shape: {esn_soc.W_in.shape}")
print(f"  W_res shape: {esn_soc.W_res.shape}")
print(f"  W_out shape: {esn_soc.W_out.shape}")

print("SOH Reservoir dimensions:")
print(f"  n_inputs: {esn_soh.n_inputs}")
print(f"  n_reservoir: {esn_soh.n_reservoir}")
print(f"  W_in shape: {esn_soh.W_in.shape}")
print(f"  W_res shape: {esn_soh.W_res.shape}")
print(f"  W_out shape: {esn_soh.W_out.shape}")

def write_array_1d(f, name, arr):
    f.write(f"const float {name}[{len(arr)}] = {{\n    ")
    for i, val in enumerate(arr):
        f.write(f"{val:.9f}f")
        if i < len(arr) - 1:
            f.write(", ")
        if (i + 1) % 6 == 0:
            f.write("\n    ")
    f.write("\n};\n\n")

def write_array_2d(f, name, arr):
    rows, cols = arr.shape
    f.write(f"const float {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        f.write("    {")
        for c in range(cols):
            f.write(f"{arr[r, c]:.9f}f")
            if c < cols - 1:
                f.write(", ")
        f.write("}")
        if r < rows - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("};\n\n")

with open(header_path, "w") as f:
    f.write("#ifndef ESN_ESTIMATOR_WEIGHTS_H\n")
    f.write("#define ESN_ESTIMATOR_WEIGHTS_H\n\n")
    f.write("// Auto-generated weights file from model_rc.pkl\n\n")
    
    f.write(f"#define ESN_N_INPUTS {esn_soc.n_inputs}\n")
    f.write(f"#define ESN_SOC_N_RESERVOIR {esn_soc.n_reservoir}\n")
    f.write(f"#define ESN_SOH_N_RESERVOIR {esn_soh.n_reservoir}\n\n")
    
    write_array_1d(f, "esn_input_means", input_means)
    write_array_1d(f, "esn_input_stds", input_stds)
    
    f.write("// SOC Weights\n")
    write_array_2d(f, "esn_soc_W_in", esn_soc.W_in)
    write_array_2d(f, "esn_soc_W_res", esn_soc.W_res)
    write_array_2d(f, "esn_soc_W_out", esn_soc.W_out)
    
    f.write("// SOH Weights\n")
    write_array_2d(f, "esn_soh_W_in", esn_soh.W_in)
    write_array_2d(f, "esn_soh_W_res", esn_soh.W_res)
    write_array_2d(f, "esn_soh_W_out", esn_soh.W_out)
    
    f.write("#endif // ESN_ESTIMATOR_WEIGHTS_H\n")

print(f"Successfully generated {header_path}")
