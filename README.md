# Workflow Energy Predictor

Predicts energy consumption and makespan for scientific workflows from pre-execution
metadata, using Random Forest models trained on WRENCH/SimGrid simulations
(energy R2 = 0.995). Includes a direct predictor app and a hybrid (gray-box) app
that generalizes to hardware unseen during training.

## The two apps

This repository contains **two** Streamlit apps:

- **`app/` - the main predictor (the model evaluated in the thesis).** It uses the
  trained Random Forest models (energy R2 = 0.995, makespan R2 = 0.916) learned from
  the main dataset of **10 workflow types simulated on 5 real SPECpower server
  configurations**. Given an uploaded workflow it predicts energy and makespan for
  each of those 5 known platforms and recommends the lowest-energy one.

- **`hybrid_app/` - an experimental gray-box variant.** A version explored as an
  extension of the thesis work. Instead of predicting energy directly, it predicts the
  platform-independent makespan and compute work and then **reconstructs** energy with
  the analytical SimGrid power model. Because the hardware specs are supplied rather
  than learned, it also works for servers **not seen during training** and adds extra
  controls: you can **add your own server** and adjust its hardware specs (cores, core
  speed, idle/busy power, and disk/network bandwidth). Disk and network bandwidth feed
  a physical data-movement term. This hybrid variant is described in the dedicated
  hybrid-model section of the thesis.

## Repository layout

```
app/         Original Streamlit app (direct Random Forest predictor)
hybrid_app/  Hybrid gray-box app (works for unseen hardware)
hybrid/      Training data for the hybrid app (full_dataset.csv, graphs_full.jsonl)
models/      Trained models (energy_model.pkl, makespan_model.pkl)
data/        Main thesis dataset (thesis_data.csv, 3,890 runs)
notebooks/   model_training.ipynb (model comparison + robustness tests)
```

## Model inspection notebook

Open `notebooks/model_training.ipynb` to see which model performs best and to run the
evaluation and robustness tests (model comparison, feature importances, recommendation
accuracy, noise/cross-workflow/interpolation experiments, and the hybrid tables).

```bash
pip install pandas numpy scikit-learn matplotlib seaborn networkx jupyter
jupyter notebook notebooks/model_training.ipynb
```

The first cell switches the working directory to the repository root, so the notebook
can be run from anywhere.

## Apps

Run either app from the repository root:

```bash
# Direct predictor (recommends among the 5 known platforms)
pip install -r app/requirements.txt
streamlit run app/app.py

# Hybrid gray-box predictor (also evaluates your own, unseen hardware)
pip install -r hybrid_app/requirements.txt
streamlit run hybrid_app/app_hybrid.py
```

Upload a WfFormat JSON workflow file (from WfCommons). The direct app predicts energy
and makespan for each of the 5 known platforms and recommends the lowest-energy one.
The hybrid app additionally reconstructs energy analytically, so you can add custom
hardware that was never part of the training set; disk and network bandwidth feed into
a physical data-movement term.

## Data

- `data/thesis_data.csv` - main dataset: 3,890 WRENCH/SimGrid runs across 10 workflow
  types and 5 platforms (7 structural + 9 platform features, energy, makespan).
- `hybrid/full_dataset.csv` + `hybrid/graphs_full.jsonl` - augmented dataset (per-task
  runtimes and DAGs) the hybrid app trains on at startup.
