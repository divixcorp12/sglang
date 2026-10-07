"""Post-run GPU/CPU clock anchor. Caller must hold cc-gpu.lock, after server shutdown."""
import argparse
import torch
from sglang.kernels.ops.moe.dspark_draft_cpu import calibrate_gpu_clock, device_module

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args=parser.parse_args()
    state=torch.zeros(6,dtype=torch.int32,device="cuda")
    result=calibrate_gpu_clock(device_module(),state,args.output)
    print({k:v for k,v in result.items() if k!="samples"})
