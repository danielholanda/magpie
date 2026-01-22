# Magpie

A lightweight, general-purpose framework for evaluating GPU kernel correctness and performance.

## Features

- **Three Evaluation Modes**: Analyze, Compare, Benchmark (WIP)
- **Heterogeneous Hardware**: AMD (HIP) and NVIDIA (CUDA) GPUs
- **Execution Environments**: Local and Container modes
- **Hardware Control**: Power and frequency management
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

```yaml
gpu:
  device_ids: [0]
  hardware:
    enabled: true

performance:
  timeout_seconds: 120
  rocprof_compute:
    args: []
  ncu:
    args: ["--target-processes", "all"]
```

### Kernel Config

See [`Magpie/kernel_config.yaml.example`](./Magpie/kernel_config.yaml.example) for full examples.

## MCP Server

MCP configuration example: [`Magpie/mcp/config.json`](./Magpie/mcp/config.json)

Available tools:
- `analyze` - Analyze kernel correctness and performance
- `compare` - Compare multiple kernel implementations
- `hardware_spec` - Query GPU hardware specifications
- `configure_gpu` - Configure GPU power and frequency

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
└── Magpie/
    ├── __init__.py          # Package initialization
    ├── __main__.py          # Entry point for python -m Magpie
    ├── main.py              # CLI implementation
    ├── config.yaml           # Framework configuration
    ├── kernel_config.yaml.example
    ├── examples/            # Example configurations
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

## Pipeline (Analyze & Compare)

```
Compiling → Correctness → Performance
    ↓            ↓             ↓
  hipcc/nvcc      testcase     rocprof-compute / ncu
```


## License

MIT License. See `LICENSE`.
