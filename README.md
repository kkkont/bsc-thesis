# Project Environment Setup

This section describes the software environment and tools used for the project. Below are the details for Python, C++, and Julia environments, including versions of key libraries and compilers.

## 1. Project Structure 

data_analysis
   |-- net_energy_consumptions.py
   |-- results
   |   |-- rq1.txt
   |   |-- rq2.txt
   |-- rq1.py
   |-- rq1_plots.py
   |-- rq2.py
   |-- shapiro_wilk.py
decision_tree
   |-- decision_tree.cpp
   |-- decision_tree.jl
   |-- decision_tree.py
experiment_data
   |-- baseline.csv
   |-- decision_tree.csv
   |-- logistic_regression.csv
   |-- merged_data.csv
   |-- naive_bayes.csv
   |-- random_forest.csv
   |-- svm.csv
fib_warmup.py
logistic_regression
   |-- logistic_regression.cpp
   |-- logistic_regression.jl
   |-- logistic_regression.py
naive_bayes
   |-- naive_bayes.cpp
   |-- naive_bayes.jl
   |-- naive_bayes.py
plots
   |-- rq1
   |   |-- accuracy_by_language.png
   |   |-- elapsed_time_by_language.png
   |   |-- net_energy_by_language.png
random_forest
   |-- random_forest.cpp
   |-- random_forest.jl
   |-- random_forest.py
svm
   |-- svm.cpp
   |-- svm.jl
   |-- svm.py

## 2. Python Environment

- **Python Version**: `Python 3.10.12`
- **Installed Packages**:
  - **`pandas`**: `2.2.3`
  - **`scikit-learn`**: `1.6.1`

## 3. C++ Environment

- **Compiler**: `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`
- **C++ Standard**: `C++17`
- **Libraries**:
  - **mlpack**: `3.4.2`
  - **Armadillo**: `10.8.2`
- **OpenMP**: `4.5`

## 4. Julia Environment

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

## 5. System Information

The measurements for the project were conducted on the following desktop system:

- **Operating System**: `Linux Ubuntu 22.04.5 LTS`
- **CPU**: `Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` (11th Gen, Quad-core)
- **RAM**: `16 GB`
- **Kernel Version**: `5.15.0-91-generic`

## 6. Performance Monitoring with `perf`

Performance measurements were conducted using the **`perf`** tool, which was configured to measure energy consumption and other relevant performance counters.

- **`perf` Configuration**:
  - Performance counters: `power/energy-pkg/`, `power/energy-cores/`, `power/energy-gpu/`, `power/energy-psys/`
