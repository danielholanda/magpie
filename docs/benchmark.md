# Benchmark Mode

Benchmark mode enables framework-level performance benchmarking for LLM inference engines (vLLM, SGLang) with integrated trace analysis capabilities.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Benchmark Mode                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────────┐   │
│  │BenchmarkConfig│  → │ BenchmarkMode │ →  │  BenchmarkResult   │   │
│  │  (YAML)       │    │               │    │  (JSON)            │   │
│  └───────────────┘    └───────────────┘    └────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Docker Container                          │   │
│  │  ┌─────────────┐        ┌─────────────────────────────────┐  │   │
│  │  │ InferenceMAX│   →    │ vLLM / SGLang Server + Client   │  │   │
│  │  │   Scripts   │        │ + Torch Profiler                │  │   │
│  │  └─────────────┘        └─────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   TraceLens Analysis (Host)                  │   │
│  │  • TraceLens_generate_perf_report_pytorch                    │   │
│  │  • TraceLens_generate_multi_rank_collective_report_pytorch   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Basic vLLM benchmark
python -m Magpie benchmark --benchmark-config examples/benchmark_vllm.yaml

# vLLM with TraceLens analysis
python -m Magpie benchmark --benchmark-config examples/benchmark_vllm_tracelens.yaml

# SGLang benchmark
python -m Magpie benchmark --benchmark-config examples/benchmark_sglang.yaml
```

## Configuration

### Minimal Example

```yaml
benchmark:
  framework: vllm              # "vllm" or "sglang"
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8               # "fp8", "fp16", "bf16"
  
  envs:
    TP: 8                      # Tensor parallelism
    CONC: 32                   # Concurrency (num_prompts = CONC * 10)
    ISL: 1024                  # Input sequence length
    OSL: 1024                  # Output sequence length
    
  profiler:
    torch_profiler:
      enabled: true            # Generate torch profiling traces
      
  timeout_seconds: 3600
```

### Full Configuration Reference

```yaml
benchmark:
  # Framework selection
  framework: vllm              # Required: "vllm" or "sglang"
  model: <model_name>          # Required: HuggingFace model name/path
  precision: fp8               # Optional: "fp8" (default), "fp16", "bf16"
  
  # Benchmark parameters
  envs:
    TP: 8                      # Tensor parallelism (GPU count)
    CONC: 32                   # Request concurrency
    ISL: 1024                  # Input sequence length
    OSL: 1024                  # Output sequence length
    RANDOM_RANGE_RATIO: 1      # Length randomization (0-1)
    MAX_MODEL_LEN: 131072      # Max model context length
    GPU_MEM_UTIL: 0.95         # GPU memory utilization (0-1)
    ENABLE_PROFILE: "true"     # Enable profiling in benchmark script
    
  # Profiler configuration
  profiler:
    # PyTorch profiler (generates JSON traces)
    torch_profiler:
      enabled: true            # Sets VLLM_TORCH_PROFILER_DIR
      
    # System profiler (rocprof-compute / ncu)
    system_profiler:
      enabled: false
      profile_args: []         # Additional profiler arguments
      
    # TraceLens trace analysis
    tracelens:
      enabled: true            # Enable TraceLens analysis
      export_format: csv       # "csv" or "excel"
      perf_report_enabled: true      # Single-rank performance report
      multi_rank_report_enabled: true # Multi-rank collective report
      gpu_arch_config: null    # Optional: GPU arch config for roofline
      
  # Execution settings
  docker_image: null           # Optional: override auto-selected image
  gpu_arch: null               # Optional: force GPU architecture
  timeout_seconds: 3600        # Benchmark timeout
  
  # Paths
  inferencemax_path: /path/to/InferenceMAX  # InferenceMAX installation
  hf_cache_path: null          # HuggingFace cache directory
  
  # InferenceMAX specific
  runner_type: mi300x          # Hardware runner type
  benchmark_script: null       # Override benchmark script
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TP` | Tensor parallelism (number of GPUs) | 1 |
| `CONC` | Request concurrency | 32 |
| `ISL` | Input sequence length | 1024 |
| `OSL` | Output sequence length | 512 |
| `RANDOM_RANGE_RATIO` | Length randomization ratio | 0.5 |
| `MAX_MODEL_LEN` | Maximum model context length | - |
| `GPU_MEM_UTIL` | GPU memory utilization | 0.95 |
| `ENABLE_PROFILE` | Enable torch profiler | "false" |

## Profiling Options

### Torch Profiler

When `torch_profiler.enabled: true`:
- Sets `VLLM_TORCH_PROFILER_DIR` automatically
- Generates JSON trace files for each GPU rank
- Traces saved to: `results/benchmark_<framework>_<timestamp>/torch_trace/`

### TraceLens Analysis

TraceLens provides automated analysis of torch profiler traces:

| Command | Description | Output |
|---------|-------------|--------|
| `TraceLens_generate_perf_report_pytorch` | Single-rank performance report | `tracelens_rank0_csvs/` |
| `TraceLens_generate_multi_rank_collective_report_pytorch` | Multi-rank collective analysis | `tracelens_collective_csvs/` |

#### TraceLens Output Files

**Single-rank report (`tracelens_rank0_csvs/`):**
- `gpu_timeline.csv` - GPU kernel timeline
- `ops_summary.csv` - Operation summary
- `ops_summary_by_category.csv` - Operations by category
- `coll_analysis.csv` - Collective communication analysis
- `kernel_summary.csv` - Kernel summary statistics

**Multi-rank collective report (`tracelens_collective_csvs/`):**
- Aggregated statistics across all GPU ranks
- Communication pattern analysis
- Load balancing metrics

## Output Structure

```
results/benchmark_vllm_<timestamp>/
├── benchmark_result.json      # Main benchmark results
├── benchmark_summary.txt      # Human-readable summary
├── server.log                 # Server logs
├── client.log                 # Client logs
├── torch_trace/               # Raw torch profiler traces
│   ├── rank-0.*.pt.trace.json.gz
│   ├── rank-1.*.pt.trace.json.gz
│   └── ...
├── tracelens_rank0_csvs/      # Single-rank TraceLens analysis
│   ├── gpu_timeline.csv
│   ├── ops_summary.csv
│   └── ...
└── tracelens_collective_csvs/ # Multi-rank TraceLens analysis
    └── ...
```

## Benchmark Result

The `benchmark_result.json` contains:

```json
{
  "task_id": "benchmark_vllm_20260211_190617",
  "framework": "vllm",
  "model": "deepseek-ai/DeepSeek-R1-0528",
  "status": "success",
  "metrics": {
    "throughput_tps": 125.5,
    "latency_p50_ms": 45.2,
    "latency_p99_ms": 128.3,
    "total_tokens": 50000,
    "duration_seconds": 398.4
  },
  "kernel_summary": [...],
  "top_bottlenecks": [...],
  "tracelens_analysis": {
    "enabled": true,
    "output_files": [...],
    "errors": []
  }
}
```

## Examples

### Quick Profiling Run

Minimal configuration for fast trace collection:

```yaml
benchmark:
  framework: vllm
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8
  
  envs:
    TP: 8
    CONC: 4                    # Small concurrency for quick run
    ISL: 128
    OSL: 64
    GPU_MEM_UTIL: 0.85
    
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      export_format: csv
      multi_rank_report_enabled: false  # Skip multi-rank for speed
      
  timeout_seconds: 1200
```

### Full Production Benchmark

```yaml
benchmark:
  framework: vllm
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8
  
  envs:
    TP: 8
    CONC: 64
    ISL: 2048
    OSL: 2048
    MAX_MODEL_LEN: 131072
    
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      export_format: csv
      perf_report_enabled: true
      multi_rank_report_enabled: true
      
  timeout_seconds: 7200
```

### SGLang Benchmark

```yaml
benchmark:
  framework: sglang
  model: meta-llama/Llama-3.1-70B-Instruct
  precision: fp16
  
  envs:
    TP: 4
    CONC: 32
    ISL: 1024
    OSL: 512
    
  profiler:
    torch_profiler:
      enabled: true
      
  timeout_seconds: 3600
```

## Troubleshooting

### Common Issues

**1. GPU Memory Error**
```
ValueError: Free memory on device (...) is less than desired GPU memory utilization
```
Solution: Reduce `GPU_MEM_UTIL` in config (e.g., 0.85)

**2. Docker Permission Error**
```
docker: permission denied
```
Solution: Add user to docker group or run with sudo

**3. TraceLens Not Found**
```
TraceLens CLI command 'TraceLens_generate_perf_report_pytorch' not found
```
Solution: TraceLens will be auto-installed. If issues persist:
```bash
pip install git+https://github.com/AMD-AIG-AIMA/TraceLens.git
```

**4. Timeout During Model Loading**

Large models (e.g., DeepSeek-R1) may need longer timeouts:
```yaml
timeout_seconds: 7200  # 2 hours
```

### Debug Mode

Enable verbose logging:
```bash
python -m Magpie benchmark --benchmark-config config.yaml --log-level DEBUG
```

## Architecture

### Components

| Component | File | Description |
|-----------|------|-------------|
| `BenchmarkMode` | `benchmarker.py` | Main orchestrator |
| `BenchmarkConfig` | `config.py` | Configuration dataclasses |
| `TraceLensAnalyzer` | `tracelens.py` | TraceLens CLI integration |
| `BenchmarkResult` | `result.py` | Result data structures |

### Execution Flow

1. **Configuration Loading**: Parse YAML config into `BenchmarkConfig`
2. **Docker Setup**: Prepare container with InferenceMAX scripts
3. **Server Launch**: Start vLLM/SGLang server inside container
4. **Client Execution**: Run benchmark client with profiling enabled
5. **Trace Collection**: Copy torch profiler traces from container
6. **TraceLens Analysis**: Run TraceLens CLI commands on host
7. **Result Generation**: Aggregate metrics and generate reports

## Related

- [TraceLens](https://github.com/AMD-AIG-AIMA/TraceLens) - Trace analysis library
- [InferenceMAX](https://github.com/AMD-AIG-AIMA/InferenceMAX) - Benchmark scripts
- [vLLM](https://github.com/vllm-project/vllm) - LLM inference engine
- [SGLang](https://github.com/sgl-project/sglang) - LLM serving framework

