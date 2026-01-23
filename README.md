# Magpie

A lightweight, general-purpose framework for evaluating GPU kernel correctness and performance.

## Features

- **Three Evaluation Modes**: Analyze, Compare, Benchmark (WIP)
- **Heterogeneous Hardware**: AMD (HIP) and NVIDIA (CUDA) GPUs
- **Execution Environments**: Local, Sandbox Container and Remote Ray Cluser
- **Hardware Control**: hardware-aware kernel evaluation under controlled execution settings
- **MCP Server**: Model Context Protocol integration for AI agents
- **Structured Reports**: JSON output for pipeline integration

## Requirements

- Python 3.10+
- AMD ROCm (HIP) or NVIDIA CUDA toolchain (for kernel compilation/profiling)
- `rocprof-compute` (AMD) or `ncu` (NVIDIA) if you enable performance profiling

## Quick Start

```bash
# Install dependencies
make install

# Analyze a kernel using a config file
python -m Magpie analyze --kernel-config Magpie/kernel_config.yaml.example

# Compare kernels directly
python -m Magpie compare kernel_v1.hip kernel_v2.hip

# Run MCP server
python -m Magpie.mcp
```

## Evaluation Modes

| Mode | Description | Status |
|------|-------------|--------|
| **Analyze** | Single kernel evaluation with testcase | ✅ |
| **Compare** | Multi-kernel comparison and ranking | ✅ |
| **Benchmark** | Performance benchmarking suite | 🚧 WIP |

## Configuration

### Framework Config (`Magpie/config.yaml`)

Key categories:
- `gpu`: force device selection and hardware control (power/frequency).
- `scheduler`: local/container/remote execution and scheduling behavior.
- `performance`: profiling and profiler configuration.
- `logging`: log levels and output formatting.

### Kernel Config

See [`Magpie/kernel_config.yaml.example`](./Magpie/kernel_config.yaml.example) for full examples.

#### Example

Example configs live in `examples/`:
- Analyze (single kernel): `examples/ck_gemm_add.yaml`
- Compare (multi-kernel): `examples/ck_grouped_gemm_compare.yaml`

## MCP Server

MCP configuration example: [`Magpie/mcp/config.json`](./Magpie/mcp/config.json)

Available tools:
- `analyze` - Analyze kernel correctness and performance
- `compare` - Compare multiple kernel implementations
- `hardware_spec` - Query GPU hardware specifications
- `configure_gpu` - Configure GPU power and frequency
- `discover_kernels` - Scan a project and suggest analyzable kernels/configs
- `suggest_optimizations` - Suggest performance optimizations from analyze output
- `create_kernel_config` - Generate a kernel config YAML for analyze

## Development

```bash
make install-dev
make lint
make format
```

## Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── Makefile
├── examples/            # Example configurations
└── Magpie/
    ├── __init__.py          # Package initialization
    ├── __main__.py          # Entry point for python -m Magpie
    ├── main.py              # CLI implementation
    ├── config.yaml           # Framework configuration
    ├── kernel_config.yaml.example
    ├── config/               # Configuration classes
    ├── core/                # Core engine components
    ├── eval/                # Evaluation pipeline
    ├── modes/               # Evaluation modes
    │   ├── analyze_eval/    # Single kernel analysis
    │   └── compare_eval/    # Multi-kernel comparison
    ├── mcp/                 # MCP Server
    │   ├── __init__.py
    │   ├── __main__.py      # Entry point for python -m Magpie.mcp
    │   ├── server.py        # MCP server implementation
    │   └── config.json       # MCP client configuration
    └── utils/               # Utility functions
```

## Overall Architecture Diagram

![Overall Architecture](docs/images/overall-architecture.png)

## Eval Pipeline

### Analyze & Compare

![Analyze & Compare Pipeline](docs/images/analyze-compare-pipeline.png)

### Benchmark

![Benchmark Pipeline](docs/images/benchmark-pipeline.png)

## License

MIT License. See `LICENSE`.
