import time
import logging
import torch
import numpy as np

# Temporarily add backend to sys.path to allow direct import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tools.biomedical_ner_tool import BiomedicalNERTool
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

# Sample text for benchmarking
SAMPLE_TEXT = "The patient was diagnosed with type 2 diabetes mellitus and coronary artery disease. They were prescribed 500mg of metformin and 10mg of atorvastatin daily. Follow-up is recommended in 3 months to monitor blood glucose and cholesterol levels."


async def benchmark():
    """Runs the full benchmark suite for the NER model."""
    logger.info("--- Starting NER Model Performance Benchmark ---")
    
    # 1. Load the model
    try:
        # Check for CUDA availability
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running benchmark on CPU. Results will be slower.")
            
        tool = BiomedicalNERTool(model_name=settings.BIOMEDICAL_NER_MODEL_NAME)
        # Manually move the pipeline to the correct device
        tool.pipe.model.to(device_str)
        
        logger.info(f"✅ NER model loaded successfully on device: {device_str.upper()}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        return

    # 2. Warmup phase
    logger.info(f"Running {WARMUP_RUNS} warmup inferences...")
    for _ in range(WARMUP_RUNS):
        try:
            _ = await tool.execute(text=SAMPLE_TEXT)
        except Exception as e:
            logger.error(f"❌ Warmup run failed: {e}")
            return
    logger.info("✅ Warmup complete.")

    # 3. Latency benchmark
    logger.info(f"Running latency benchmark ({BENCHMARK_RUNS} runs)...")
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        try:
            _ = await tool.execute(text=SAMPLE_TEXT)
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

    # 4. Throughput benchmark
    logger.info(f"Running throughput benchmark...")
    # Create a dataset for batch processing
    THROUGHPUT_SAMPLES = 500
    throughput_dataset = [SAMPLE_TEXT] * THROUGHPUT_SAMPLES
    
    start_time = time.perf_counter()
    # Process the entire dataset as a batch
    _ = tool.pipe(throughput_dataset, batch_size=32)
    end_time = time.perf_counter()
    
    total_duration = end_time - start_time
    throughput = THROUGHPUT_SAMPLES / total_duration if total_duration > 0 else 0
    logger.info("✅ Throughput benchmark complete.")

    # 5. Print results
    print("\n" + "="*40)
    print("      NER Model Inference Rate Results")
    print("="*40)
    print(f"Device Used:            {device_str.upper()}")
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
    import asyncio
    asyncio.run(benchmark())
