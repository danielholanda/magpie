# AIG-Kernel-Eval

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
python main.py analyze -k kernel_config.yaml

# Compare kernels
python main.py compare kernel_v1.hip kernel_v2.hip
```

## Evaluation Modes

| Mode | Description | Status |
|------|-------------|--------|
| **Analyze** | Single kernel evaluation with testcase | ✅ |
| **Compare** | Multi-kernel comparison and ranking | ✅ |
| **Benchmark** | Performance benchmarking suite | 🚧 WIP |

## Configuration

### Framework Config (`config.yaml`)

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

See [`kernel_config.yaml.example`](./kernel_config.yaml.example) for full examples.

## MCP Server

MCP configuration example: [`mcp_config.json`](./mcp_config.json)

Available tools:
- `analyze` - Analyze kernel correctness and performance
- `compare` - Compare multiple kernel implementations
- `hardware_spec` - Query GPU hardware specifications
- `configure_gpu` - Configure GPU power and frequency

## Project Structure

```
├── config.yaml              # Framework configuration
├── main.py                  # CLI entry point
├── src/
│   ├── config/              # Configuration classes
│   ├── eval/                # Evaluation pipeline
│   │   ├── compiling.py     # Kernel compilation
│   │   ├── correctness.py   # Correctness verification
│   │   └── performance.py   # Performance profiling
│   ├── modes/               # Evaluation modes
│   │   ├── analyze_eval/    # Single kernel analysis
│   │   └── compare_eval/    # Multi-kernel comparison
│   └── mcp/                 # MCP server
```

## Pipeline (Analyze & Compare)

```
Compiling → Correctness → Performance
    ↓            ↓             ↓
  hipcc      testcase     rocprof-compute / ncu
```


## License

MIT
