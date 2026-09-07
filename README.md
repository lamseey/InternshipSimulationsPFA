# FAR and FedFDP Federated Learning Simulations

This repository contains a Jupyter notebook that compares fairness-aware and robust federated learning methods on non-IID image-classification data.

## Notebook

[`InternshipSimulations.ipynb`](InternshipSimulations.ipynb) evaluates:

- FAR-CM(NNM), with fairness and robustness aggregation regimes
- FedAvg and q-FFL baselines
- FedFair without differential privacy
- FedFDP with fairness-aware clipping and differential privacy
- Byzantine attacks and differential-privacy sweeps
- MNIST and Fashion-MNIST datasets

Experiments report accuracy, client-level fairness metrics, robustness results, and the FedFDP balanced performance fairness metric ($\Psi$).

## Requirements

- Python 3.9 or newer
- PyTorch 2.0 or newer
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the main dependencies with:

```bash
python -m pip install torch torchvision numpy matplotlib notebook
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Running the simulations

1. Open `InternshipSimulations.ipynb` in Jupyter or VS Code.
2. Run the cells from top to bottom.
3. Allow the notebook to download MNIST and Fashion-MNIST the first time they are used.
4. Review the generated logs, tables, and plots in the experiment sections.

The default configuration runs multiple random seeds (`SEEDS`) and is intended for reproducible multi-seed evaluation. Full experiments can take a while, especially on CPU.

## Configuration

The Configuration cell controls the main runtime and experiment settings, including:

- `DATASET`: `mnist` or `fashion_mnist`
- `N_ROUNDS` and `LOCAL_EPOCHS`: federated training duration
- `SEEDS`: random seeds; at least two are required
- `DEVICE`: CPU or CUDA device selection
- `SUBSAMPLE_TRAIN` and `SUBSAMPLE_TEST`: optional dataset-size limits for quicker runs
- `FAR_ALPHAS`: fairness/robustness aggregation sweep
- DP parameters such as epsilon and delta

Use smaller round counts or dataset subsampling for a quick smoke test. Restore the original settings for the full evaluation.

## Reproducibility

Each experiment runs independently for every seed and reports aggregate results across seeds. Dataset partitions and model configuration are recorded by the notebook so results can be compared across runs. Network access is required for the initial dataset download.

## Project structure

```text
.
├── InternshipSimulations.ipynb
└── README.md
```
