# ADR-0002: Test and micro-benchmark frameworks

- Status: Accepted
- Date: 2026-05-30
- Deciders: Akmal Khujdarov

## Context

The project needs C++ unit and correctness tests wired into CTest, Python tests for the bindings and the HTTP API, and kernel-level micro-benchmarks. CI runs the C++ and Python suites on every change.

## Decision

- C++ tests: GoogleTest, with GoogleMock available for fakes. Fetched via CMake FetchContent and registered with CTest.
- Python tests: pytest.
- C++ micro-benchmarks: Google Benchmark, for kernel timing.

## Consequences

- GoogleTest brings death tests, rich matchers, and gmock, and integrates cleanly with CTest and CI.
- FetchContent pins the dependency versions in-tree without a system install.
- Cost: a heavier dependency than a header-only framework, and a fetch on the first configure.

## Alternatives considered

- Catch2. Header-only with pleasant syntax, but GoogleTest's maturity, mocking support, and ubiquity in CI won out.
- doctest. Fast to compile, but the smallest ecosystem of the three.
