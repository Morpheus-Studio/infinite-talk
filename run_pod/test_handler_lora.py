#!/usr/bin/env python3
"""
Test script for LoRA runpod_handler.py
"""
import json
from runpod_handler import handler

# Load test input
with open('test_input_lora.json', 'r') as f:
    test_job = json.load(f)

# Add a mock job ID
test_job['id'] = 'test-job-lora-001'

print("=" * 60)
print("Testing RunPod Handler (LoRA)")
print("=" * 60)
print("\nInput:")
print(json.dumps(test_job, indent=2))
print("\n" + "=" * 60)
print("Running handler...")
print("=" * 60 + "\n")

# Run the handler
result = handler(test_job)

print("\n" + "=" * 60)
print("Result:")
print("=" * 60)
print(json.dumps(result, indent=2))
print("=" * 60)
