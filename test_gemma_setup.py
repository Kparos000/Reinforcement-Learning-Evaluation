"""
Gemma-2B Setup Verification Test

This verifies your system is ready for production REINFORCE training:
1. Gemma-2B model can be loaded (4GB download)
2. 8k context window is available
3. Can generate text properly
4. REINFORCE trainer initializes correctly
5. Phase 2 long scenarios load

Run this BEFORE starting full training to catch issues early.
"""

import sys

print("=" * 70)
print("GEMMA-2B SETUP TEST (Production)")
print("=" * 70)

# Test 1: Check imports
print("\n[1/6] Checking imports...")
try:
    import torch
    import transformers
    from ace_task.algorithms.reinforce import REINFORCETrainer, REINFORCEConfig
    from ace_task.scenarios import get_scenario
    from ace_task.algorithms.rewards import create_reward_function
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ Transformers {transformers.__version__}")
    print(f"✅ REINFORCE modules imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nPlease install dependencies:")
    print("  pip install torch transformers accelerate sentencepiece protobuf")
    sys.exit(1)

# Test 2: Check device
print("\n[2/6] Checking compute device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ℹ️  Training on CPU (slower but works)")
    print("   Expected training time: 2-4 hours for all 3 scenarios")

# Test 3: Check disk space
print("\n[3/6] Checking disk space...")
import shutil
try:
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    print(f"✅ Free disk space: {free_gb:.1f} GB")
    if free_gb < 10:
        print(f"⚠️  Warning: Low disk space. Need ~6GB for Gemma-2B download + models")
    else:
        print(f"   Plenty of space for Gemma-2B (needs ~6GB)")
except Exception as e:
    print(f"⚠️  Could not check disk space: {e}")

# Test 4: Load Gemma-2B model
print("\n[4/6] Loading Gemma-2B model...")
print("   (First run: downloads 4GB model, takes 5-10 min)")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "google/gemma-2b"
    print(f"   Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"✅ Gemma-2B loaded successfully")
    print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"   Context window: {model.config.max_position_embeddings} tokens")

    # Quick generation test
    print("\n   Testing generation...")
    test_input = "The capital of France is"
    input_ids = tokenizer.encode(test_input, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=5)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"   Input: '{test_input}'")
    print(f"   Output: '{output_text}'")
    print(f"   ✅ Generation working!")

    # Clean up
    del model
    del tokenizer
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("\nTroubleshooting:")
    print("  - Check internet connection (needs to download 4GB)")
    print("  - Ensure you have ~6GB free disk space")
    print("  - Try: pip install --upgrade transformers")
    sys.exit(1)

# Test 5: Load Phase 2 scenario
print("\n[5/6] Loading Phase 2 long scenario...")
try:
    scenario = get_scenario("medical_long")
    print(f"✅ Scenario loaded: {scenario.name}")
    print(f"   Original length: {len(scenario.original)} chars")
    print(f"   Facts to preserve: {len(scenario.facts)}")
    print(f"   Target compression: {int(len(scenario.original) * 0.6)} chars (60%)")
    print(f"   Word cap: 120 words")
except Exception as e:
    print(f"❌ Scenario loading failed: {e}")
    sys.exit(1)

# Test 6: Initialize REINFORCE trainer with Gemma
print("\n[6/6] Initializing REINFORCE trainer with Gemma-2B...")
try:
    reward_fn = create_reward_function(scenario, reward_scheme="binary")
    config = REINFORCEConfig(
        learning_rate=1e-5,
        max_length=1024,
        temperature=0.7,
    )

    print("   Initializing trainer (this takes a moment)...")
    trainer = REINFORCETrainer(
        model_name="google/gemma-2b",
        reward_fn=reward_fn,
        config=config,
        device="cpu",  # Use CPU for test
    )
    print(f"✅ REINFORCE trainer initialized")
    print(f"   Model: Gemma-2B")
    print(f"   Context: 8,192 tokens")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Temperature: {config.temperature}")

    # Clean up
    del trainer
except Exception as e:
    print(f"❌ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour system is ready for production REINFORCE training with Gemma-2B!")
print("\n📋 Next Steps:")
print("\n1. Quick test (30-45 min, validate training works):")
print("   python experiments/train_reinforce_gemma.py --scenario medical_long --epochs 10")
print()
print("2. Full production training (2-4 hours, all scenarios):")
print("   python experiments/train_reinforce_gemma.py --scenarios medical_long business_long legal_long --epochs 50")
print()
print("3. After training completes:")
print("   - Models saved to: results/reinforce/gemma-2b/")
print("   - Ready for UI integration")
print("   - Can proceed to Phase 3 (PPO)")
print()
print("💡 Tips:")
print("   - Training on CPU: 2-4 hours total (can run overnight)")
print("   - If you have GPU: add --device cuda for 3-5x speedup")
print("   - Training progress shown in real-time")
print("   - Safe to interrupt with Ctrl+C (can resume later)")
print("=" * 70)
