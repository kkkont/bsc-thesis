# Project Environment Setup

This section describes the software environment and tools used for the project. Below are the details for Python, C++, and Julia environments, including versions of key libraries and compilers.

## 1. Python Environment

- **Python Version**: `Python 3.10.12`
- **Installed Packages**:
  - **`pandas`**: `2.2.3`
  - **`scikit-learn`**: `1.6.1`

## 2. C++ Environment

- **Compiler**: `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`
- **C++ Standard**: `C++17`
- **Libraries**:
  - **mlpack**: `3.4.2`
  - **Armadillo**: `10.8.2`
- **OpenMP**: `4.5`

## 3. Julia Environment

- **Julia Version**: `1.11.3`
- **Installed Packages**:
  - **CSV**: `v0.10.15`
  - **DataFrames**: `v1.7.0`
  - **MLJ**: `v0.20.7`
  - **MLJBase**: `v1.7.0`
  - **MLJDecisionTreeInterface**: `v0.4.2`
  - **MLJLinearModels**: `v0.10.0`
  - **MLJModels**: `v0.17.8`
  - **MLJNaiveBayesInterface**: `v0.1.6`
  - **MLJScikitLearnInterface**: `v0.7.0`

## 4. System Information

The measurements for the project were conducted on the following desktop system:

- **Operating System**: `Linux Ubuntu 22.04.5 LTS`
- **CPU**: `Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` (11th Gen, Quad-core)
- **RAM**: `16 GB`
- **Kernel Version**: `5.15.0-91-generic`

## 5. Performance Monitoring with `perf`

Performance measurements were conducted using the **`perf`** tool, which was configured to measure energy consumption and other relevant performance counters.

- **`perf` Configuration**:
  - Performance counters: `power/energy-pkg/`, `power/energy-cores/`, `power/energy-gpu/`, `power/energy-psys/`
