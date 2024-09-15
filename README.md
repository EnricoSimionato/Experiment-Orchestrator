# Experiment Orchestrator

This repository provides a versatile framework for orchestrating and running machine learning experiments, with a focus on simplifying configuration management, experiment execution, and analysis, particularly for large language models (LLMs) and classification tasks.

## Features

- **Modular design**: Support for multiple experiment types, such as training new model architectures, benchmarking models, and performing specialized analyses on LLMs.
- **Configuration-driven**: Easily configure experiments through external configuration files.
- **Utilities for common ML tasks**: Includes utilities for causal language modeling, classification, and plotting, making it easier to build, train, and evaluate models.
- **Custom plotting and storage utilities**: Helpful tools for visualizing results and managing experiment data.

---

## Directory Structure

The project is organized as follows:

```
src/
├── exporch/                           
│   ├── configuration/                 # Handles experiment configurations
│   │   ├── __init__.py
│   │   └── config.py                  # Configuration handler
│   ├── experiment/                    # Experiment management
│   │   ├── __init__.py
│   │   └── benchmarking_experiment.py # Script for benchmarking models
│   │   └── experiment.py              # Core experiment logic
├── utils/                             # Utility functions for various tasks
│   ├── __init__.py
│   ├── causal_language_modeling/      # Utilities for causal language modeling tasks
│   │   ├── __init__.py
│   │   ├── conversation_utils.py      # Tools for handling conversational data
│   │   ├── pl_datasets.py             # Datasets handling for language modeling
│   │   ├── pl_models.py               # Models specific to language modeling
│   │   ├── pl_trainer.py              # Trainer setup for language modeling
│   ├── classification/                # Utilities for classification tasks
│   │   ├── __init__.py
│   │   ├── classification_utils.py    # Helper functions for classification
│   │   ├── pl_datasets.py             # Dataset handling for classification
│   │   ├── pl_metrics.py              # Classification metrics tools
│   │   ├── pl_models.py               # Models specific to classification
│   │   ├── pl_trainer.py              # Trainer setup for classification
│   ├── device_utils/                  # Device management utilities (e.g., GPU)
│   │   ├── __init__.py
│   │   └── device_utils.py            # Utilities for handling devices (GPU, etc.)
│   ├── pl_utils/                      # General PyTorch Lightning utilities
│   │   ├── __init__.py
│   │   ├── pl_datasets.py             # PyTorch Lightning dataset utilities
│   │   └── utility_mappings.py        # Mappings for utility functions
│   ├── plot_utils/                    # Plotting utilities for visualizations
│   │   ├── __init__.py
│   │   └── heatmap.py                 # Plotting heatmaps for experiment results
│   ├── print_utils/                   # Printing utilities for console outputs
│   │   ├── __init__.py
│   │   └── print_utils.py             # Helper functions for pretty printing
│   ├── storage_utils/                 # Storage utilities for saving/loading data
│   │   ├── __init__.py
│   │   └── storage_utils.py           # Tools for storing experiment results
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
└── requirements.txt                   # Python package dependencies
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/experiment-orchestrator.git
   cd experiment-orchestrator
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

The `config.py` file located in `src/exporch/configuration/` is responsible for managing the experiment setup. You can specify hyperparameters, dataset paths, and other experiment settings here.

### Running Experiments

To run an experiment, use the scripts in `src/exporch/experiment/`:
- **Benchmarking Experiment**: This script allows you to benchmark different model configurations and architectures.
  
  ```bash
  python src/exporch/experiment/benchmarking_experiment.py
  ```

- **Custom Experiment**: Define and run custom experiments by modifying `experiment.py` according to your needs.

### Utilities

Several utility modules are available in the `utils/` directory:
- **causal_language_modeling**: Utilities to help with language model training, dataset handling, and evaluation.
- **classification**: Functions to assist with classification tasks, including dataset loading, metrics computation, and model handling.
- **plot_utils**: Heatmap generation and other plotting utilities to visualize your experiment results.
- **print_utils**: Functions to print data in a formatted manner for debugging and analysis.
- **storage_utils**: Save and load experiment data easily using these utility functions.

