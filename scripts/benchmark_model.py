import time
import logging
import torch
from PIL import Image
import numpy as np
import asyncio # Import asyncio

# Temporarily add backend to sys.path to allow direct import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tools.skin_disease_prediction_tool import SkinDiseasePredictionTool
from backend.config import settings

# --- Constants ---
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
THROUGHPUT_DURATION_SECS = 5

# --- Setup ---
# Suppress noisy logs from other modules
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Benchmark")
logger.setLevel(logging.INFO)

def create_dummy_image(width=224, height=224) -> Image.Image:
    """Creates a random PIL Image."""
    return Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))

async def benchmark(): # Make the function async
    """Runs the full benchmark suite for the skin disease prediction model."""
    logger.info("--- Starting Model Performance Benchmark ---")
    
    # 1. Load the model
    try:
        tool = SkinDiseasePredictionTool(
            model_path=settings.DISEASE_PREDICTION_MODEL,
            class_names=settings.SKIN_DISEASE_CLASS_NAMES
        )
        logger.info(f"✅ Model loaded successfully on device: {tool.device.upper()}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # 2. Prepare dummy input
    dummy_image = create_dummy_image()

    # 3. Warmup phase
    logger.info(f"Running {WARMUP_RUNS} warmup inferences...")
    for _ in range(WARMUP_RUNS):
        try:
            _ = await tool.execute(image=dummy_image) # Use await and correct method name
        except Exception as e:
            logger.error(f"❌ Warmup run failed: {e}")
            return
    logger.info("✅ Warmup complete.")

    # 4. Latency benchmark
    logger.info(f"Running latency benchmark ({BENCHMARK_RUNS} runs)...")
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        try:
            _ = await tool.execute(image=dummy_image) # Use await and correct method name
        except Exception as e:
            logger.error(f"❌ Latency benchmark run failed: {e}")
            continue
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
    
    if not latencies:
        logger.error("❌ No successful latency benchmark runs.")
        avg_latency = p95_latency = p99_latency = min_latency = max_latency = float('nan')
    else:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = min(latencies)
        max_latency = max(latencies)
    
    logger.info("✅ Latency benchmark complete.")

    # 5. Throughput benchmark
    logger.info(f"Running throughput benchmark ({THROUGHPUT_DURATION_SECS} seconds)...")
    inference_count = 0
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < THROUGHPUT_DURATION_SECS:
        try:
            _ = await tool.execute(image=dummy_image) # Use await and correct method name
            inference_count += 1
        except Exception as e:
            logger.error(f"❌ Throughput benchmark run failed: {e}")
            break
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    throughput = inference_count / total_duration if total_duration > 0 else 0
    logger.info("✅ Throughput benchmark complete.")

    # 6. Print results
    print("\n" + "="*40)
    print("      Model Inference Rate Results")
    print("="*40)
    print(f"Device Used:            {tool.device.upper()}")
    print("-" * 40)
    print("Latency:")
    print(f"  - Average:              {avg_latency:.2f} ms")
    print(f"  - p95:                  {p95_latency:.2f} ms")
    print(f"  - p99:                  {p99_latency:.2f} ms")
    print(f"  - Min / Max:            {min_latency:.2f} ms / {max_latency:.2f} ms")
    print("-" * 40)
    print("Throughput:")
    print(f"  - Inferences / Second:  {throughput:.2f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running benchmark on CPU. Results will be slower.")
    
    asyncio.run(benchmark()) # Run the async benchmark function
