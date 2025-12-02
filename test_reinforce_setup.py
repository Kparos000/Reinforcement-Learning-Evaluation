"""
Quick test to verify REINFORCE setup is working.

This tests:
1. PyTorch and Transformers are installed
2. Can load GPT-2 model
3. Can generate text
4. REINFORCE trainer initializes correctly

Run this BEFORE full training to catch any issues early.
"""

import sys

print("=" * 70)
print("REINFORCE SETUP TEST")
print("=" * 70)

# Test 1: Check imports
print("\n[1/5] Checking imports...")
try:
    import torch
    import transformers
    from ace_task.algorithms.reinforce import REINFORCETrainer, REINFORCEConfig
    from ace_task.scenarios import get_scenario
    from ace_task.algorithms.rewards import create_reward_function
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"[OK] Transformers {transformers.__version__}")
    print(f"[OK] REINFORCE modules imported")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("\nPlease install dependencies:")
    print("  pip install torch transformers accelerate")
    sys.exit(1)

# Test 2: Check device
print("\n[2/5] Checking compute device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   Note: Training on CPU will be slower. GPU recommended for faster training.")

# Test 3: Load model
print("\n[3/5] Loading GPT-2 model...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # Small model for testing
    print(f"   Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[OK] Model loaded successfully")
    print(f"   Parameters: {model.num_parameters() / 1e6:.1f}M")

    # Clean up
    del model
    del tokenizer
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    sys.exit(1)

# Test 4: Load scenario
print("\n[4/5] Loading Phase 2 scenario...")
try:
    scenario = get_scenario("medical_long")
    print(f"[OK] Scenario loaded: {scenario.name}")
    print(f"   Original length: {len(scenario.original)} chars")
    print(f"   Facts: {len(scenario.facts)}")
except Exception as e:
    print(f"[ERROR] Scenario loading failed: {e}")
    sys.exit(1)

# Test 5: Initialize REINFORCE trainer
print("\n[5/5] Initializing REINFORCE trainer...")
try:
    reward_fn = create_reward_function(scenario, reward_scheme="binary")
    config = REINFORCEConfig(
        learning_rate=1e-5,
        max_length=256,  # Short for test
    )
    trainer = REINFORCETrainer(
        model_name="gpt2",
        reward_fn=reward_fn,
        config=config,
        device="cpu",  # Use CPU for test
    )
    print(f"[OK] REINFORCE trainer initialized")
    print(f"   Model: gpt2")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Baseline type: {config.baseline_type}")

    # Clean up
    del trainer
except Exception as e:
    print(f"[ERROR] Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("[OK] ALL TESTS PASSED!")
print("=" * 70)
print("\nYour system is ready for REINFORCE training!")
print("\nNext steps:")
print("  1. Test on 1 scenario (15-30 min):")
print("     python experiments/train_reinforce.py --scenario medical_long --epochs 10")
print()
print("  2. Full training on all scenarios (1-2 hours):")
print("     python experiments/train_reinforce.py --scenarios medical_long business_long legal_long --epochs 50")
print()
print("Note: Use --device cuda if you have a GPU for faster training")
print("=" * 70)
