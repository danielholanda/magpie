# Magpie

A lightweight, general-purpose framework for evaluating GPU kernel correctness and performance.

## Features

- **Three Evaluation Modes**: Analyze, Compare, Benchmark (WIP)
- **Heterogeneous Hardware**: AMD (HIP) and NVIDIA (CUDA) GPUs
- **Execution Environments**: Local and Sandbox modes
- **Hardware Control**: Power and frequency management
- **MCP Server**: Model Context Protocol integration for AI agents
- **Structured Reports**: JSON output for pipeline integration

## Quick Start

```bash
# Install
make install

# Analyze a kernel
python -m Magpie analyze -k Magpie/kernel_config.yaml

# Compare kernels
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
    │   ├── compiling.py     # Kernel compilation
    │   ├── correctness.py   # Correctness verification
    │   └── performance.py   # Performance profiling
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
  hipcc      testcase     rocprof-compute / ncu
```


## License

MIT
