# Architecture Decision Records

Each ADR captures one significant decision: its context, the choice, and the consequences. The format follows a lightweight MADR style; see [template.md](template.md).

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-cpp-python-split.md) | C++/CUDA core with a Python frontend | Accepted |
| [0002](0002-test-frameworks.md) | Test and micro-benchmark frameworks | Accepted |
| [0003](0003-dependency-boundary.md) | Dependency boundary and "from scratch" scope | Accepted |

Planned: device backend abstraction (CPU reference vs CUDA), KV-cache block size, quantization scheme, speculative-decoding draft choice.
