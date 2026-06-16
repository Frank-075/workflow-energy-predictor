# Interactive Hybrid Energy Estimator (gray-box)

A Streamlit app that estimates the energy of a scientific workflow on **any**
platform, including hardware that was never part of the training set. It is the
interactive companion to the hybrid (gray-box) model described in the thesis
(`hybrid/`).

## How it works

Instead of predicting energy directly (which scales with platform power and so
cannot be extrapolated to unseen hardware), the app predicts two
platform-independent quantities and recombines them with the analytical SimGrid
power model:

```
makespan  T   <- Random Forest on workflow metadata + platform   (transfers across hardware)
work / FLOPs  <- Random Forest on workflow metadata only          (platform-independent)

E = num_hosts * watt_idle * T
    + (watt_busy - watt_idle) * (work / (speed_gf * 1e9)) / cores_per_host
```

Because the power terms (`watt_idle`, `watt_busy`, `speed_gf`, `cores_per_host`)
are supplied explicitly rather than learned, the estimate generalizes to
platforms outside the training set.

## Using the app

1. **Upload a WfFormat workflow JSON.** The seven structural features are
   extracted automatically, and the workflow type is read from the JSON `name`
   field (e.g. `"Genome-synthetic-instance"` -> `Genome`).
2. **Enter the hardware specification.** Start from one of the five study
   platforms or pick *Custom platform* and type in the cores, core speed
   (GFlops), and idle/busy power of a platform that does not appear in training.
3. **Click *Estimate energy*** to get predicted energy, makespan, and work.

## Run

```bash
pip install -r requirements.txt
streamlit run app_hybrid.py
```

The two Random Forest models are trained once on first load (cached for the
session) from `../hybrid/full_dataset.csv` and `../hybrid/graphs_full.jsonl`
(2,030 runs, 5 workflow types, 5 platforms). Both use `n_estimators=300`,
`random_state=42`, and log10 targets, matching `hybrid/hybrid_v2.py`.

## Relation to the main app

`../app.py` is the main tool: it predicts energy directly for the five study
platforms and recommends the lowest-energy one. This hybrid app trades that
turn-key convenience for the ability to evaluate **arbitrary, unseen** hardware.
