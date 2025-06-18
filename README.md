# Burke, Hsiang, and Miguel 2015 - Python Replication

This repository contains a Python implementation of the replication code for Burke, Hsiang, and Miguel (2015).

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data/`: Contains input and output data
  - `input/`: Original data files
  - `output/`: Generated data files
- `data_processing/`: Scripts for data preparation
- `projections/`: Scripts for computing projections
- `figures/`: Scripts for generating figures
- `utils/`: Utility functions and helper code

## Running the Replication

To run the full replication:
```bash
python run_full_replication.py
```

To run individual components, see the documentation in each module.

## Original Paper

This is a replication of Burke, Hsiang, and Miguel (2015). Please cite the original paper when using this code.
