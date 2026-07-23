#!/usr/bin/env python3
"""
ref_dump.py — Dump ALL intermediate tensors from the known-good PyTorch Qwen3.5-0.8B forward pass.

Captures every intermediate tensor (inputs/outputs of every submodule) and saves them
to a safetensors file for differential testing against our TTIR runtime.

Usage:
  source /tmp/refenv/bin/activate
  python3 prototypes/ref_dump.py --prompt "Hello" --output /tmp/ref_tensors.safetensors

The script runs on CPU (no GPU needed). The model is only 0.8B params.
"""
import argparse
import os
import sys
import hashlib
import torch
import numpy as np
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="/tmp/ref_tensors.safetensors")
    parser.add_argument("--layers", type=int, default=24)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Tokenizing: '{args.prompt}'")
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]
    print(f"  tokens: {input_ids.tolist()}  ({input_ids.shape[1]} tokens)")

    print(f"Loading model {args.model} on CPU (bf16)...")
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    # ── Register hooks on ALL submodules ──
    captured = {}
    
    def make_hook(name):
        def hook_fn(module, inp, out):
            # Capture input (first tensor) and output
            if isinstance(inp, tuple) and len(inp) > 0:
                inp_tensor = inp[0]
                if isinstance(inp_tensor, torch.Tensor):
                    captured[f"{name}__in"] = inp_tensor.detach().float().contiguous()
            if isinstance(out, torch.Tensor):
                captured[f"{name}__out"] = out.detach().float().contiguous()
            elif isinstance(out, tuple) and len(out) > 0:
                if isinstance(out[0], torch.Tensor):
                    captured[f"{name}__out"] = out[0].detach().float().contiguous()
        return hook_fn

    # Register on every named module
    hook_count = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        module.register_forward_hook(make_hook(name))
        hook_count += 1
    print(f"  Registered {hook_count} hooks")

    # ── Run forward pass ──
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    logits = outputs.logits
    print(f"  logits shape: {logits.shape}")
    
    # Get the predicted token
    next_token = logits[0, -1].argmax().item()
    print(f"  predicted next token: {next_token} ({tokenizer.decode([next_token])})")
    
    # Also capture the logits and final hidden state
    captured["logits"] = logits.detach().float().contiguous()
    captured["input_ids"] = input_ids.detach().float().contiguous()
    
    # ── Save to safetensors ──
    # Flatten all tensors to 1D for safetensors (store shape in name)
    flat_tensors = {}
    shapes = {}
    for name, tensor in captured.items():
        flat = tensor.reshape(-1).contiguous()
        flat_tensors[name] = flat
        shapes[name] = list(tensor.shape)
    
    print(f"\nCaptured {len(flat_tensors)} tensors:")
    
    # Print summary table
    print(f"{'Tensor':<70} {'Shape':<25} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print("-" * 130)
    for name in sorted(flat_tensors.keys()):
        t = captured[name]
        shape_str = str(shapes[name])
        print(f"{name:<70} {shape_str:<25} {t.mean().item():>10.4f} {t.std().item():>10.4f} {t.abs().max().item():>10.4f}")
    
    # Save
    save_file(flat_tensors, args.output)
    
    # Save shapes metadata
    import json
    shapes_path = args.output.replace(".safetensors", "_shapes.json")
    with open(shapes_path, "w") as f:
        json.dump(shapes, f, indent=2)
    
    print(f"\nSaved {len(flat_tensors)} tensors to {args.output}")
    print(f"Saved shapes to {shapes_path}")
    print(f"\nNext token: {next_token} ({tokenizer.decode([next_token])})")

if __name__ == "__main__":
    main()
