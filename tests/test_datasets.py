import datasets
from datasets import load_dataset
import pytest
from huggingface_hub import login


def test_load_ZebraLogicBench():
    try:
        dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    except Exception as e:
        pytest.fail(f"Loading dataset failed with exception: {e}")
    
