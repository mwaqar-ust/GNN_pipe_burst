# Graph Neural Networks for Passive Transient Source Localisation

## Start on Windows

If your previous GNN tutorial already runs, you can open this notebook with its existing working kernel. The dependencies are unchanged. For a separate installation or a new computer, follow the steps below.

1. **Extract the ZIP completely.** Find the folder containing `setup_windows.bat`, `environment.yml` and `IAHR_GNN_HKUST_Tutorial.ipynb` together. Copy that folder's path from File Explorer.
2. Open **Anaconda Prompt**, enter that folder with `cd /d "your copied folder path"`, then run:

   ```bat
   setup_windows.bat
   ```

   For example, if those files are directly in Documents:

   ```bat
   cd /d "%USERPROFILE%\Documents\IAHR_GNN_HKUST_Tutorial"
   dir environment.yml
   setup_windows.bat
   ```

   If `environment.yml` is missing, open the inner extracted project folder and use its actual path.

3. After setup succeeds:

   ```bat
   conda activate iahr-gnn-hkust-tutorial
   jupyter lab
   ```

4. Open **`IAHR_GNN_HKUST_Tutorial.ipynb`**, select **Python (IAHR GNN HKUST Tutorial)**, and choose **Kernel → Restart Kernel and Run All Cells**.

The new environment is named **`iahr-gnn-hkust-tutorial`**. Installation checks the NVIDIA hardware/driver and selects a compatible official PyTorch wheel. A real tensor calculation verifies the device; CPU execution remains available. RTX 3070 and RTX 4060 are covered by the selection logic. Internet is needed for the first dependency installation; prepared examples then run locally.

To choose CPU explicitly, run `setup_windows.bat --cpu`. Linux users can run `bash setup_unix.sh`. Detailed instructions and troubleshooting are in [docs/SETUP.md](docs/SETUP.md) and [docs/GPU_SETUP.md](docs/GPU_SETUP.md).

## Follow the story

| Part | What you inspect |
|---|---|
| The mystery | Network, sensors, pressure record and hidden source |
| Graph language | Nodes, edges, degree, paths, weights and adjacency |
| First-arrival physics | Graph travel times overlaid on independently simulated signals |
| Uncertain observations | Background pressure noise, repicked arrivals and additional timing error |
| Learned representations | Input features, embeddings and neighbour aggregation |
| The GNN | Two residual message-passing layers and shared pipe outputs |
| Training | Graph-derived timing examples, separate source splits, loss and validation |
| Localisation | Predicted pipe, within-pipe fraction and network-distance error |
| Exploration | Sensor layouts, ambiguity and an optional new local TSNet burst |

Each main topic has saved static figures. Expand **Explore interactively** for live controls and sub-tabs after its cell runs. The model uses arrival-time features. Raw signals provide the observations and explain the uncertainty in those features.

The HKUST case has 31 nodes and 33 pipes, including the fixed-head supply T1. All pipes remain in the graph. The default source-domain threshold of 100 m permits bursts in 21 original pipes, with ten junction sensors. The controlled hydraulic setup uses a 40 m supply head, zero elevations and initial demands, and diameter-dependent wave speeds.

## Retrain or change the configuration

The controls are in the first code cell, under **0. Prepare the experiment**:

```python
RETRAIN = True
TRAINING_EPOCHS = 500
DEVICE = 'auto'
```

Restart the kernel and run all cells. The setup fits the model once, restores its best validation checkpoint and uses that model throughout the notebook. Early stopping can finish before the maximum epoch count.

- The supplied model stays at `models/gnn_demo.npz`.
- Your newly trained model is saved at `outputs/gnn_user.npz`.
- Completed training epochs are checkpointed at `outputs/training_last.pt`.
- `SENSOR_NODES=None` selects the supplied ten-sensor layout. A new layout or source-length threshold needs retraining.
- Training uses 10 ms Gaussian timestamp uncertainty and 100 Hz quantisation by default. Change the controls and retrain to teach another training distribution.
- Changing source counts or split seed also requires a corresponding hydraulic test set, so the cached tests remain independent.

The notebook prints the active device, completion time and training progress. GPU speed varies by hardware; a small graph is also practical on CPU. To resume an interrupted training session, keep the same configuration and set the training call's `resume=True`; its checkpoint path must be the existing `outputs/training_last.pt`.

## Prepared data and interpretation

Training, validation and held-out graph examples use distinct source locations. The hydraulic cache was generated separately with TSNet. Arrival picking is rerun on its stored 100 Hz signals. The original source junction is excluded from the model's input graph.

The uncertainty controls distinguish pressure noise from additional timestamp jitter. Missing picks remain missing and are shown explicitly. The results display pipe accuracy and mean network-distance error; the latter includes wrong-pipe estimates. When an event cannot be localised, no distance is invented. Read the coverage alongside the table.

See [docs/SCIENTIFIC_NOTES.md](docs/SCIENTIFIC_NOTES.md) for assumptions and [docs/TEACHING_GUIDE.md](docs/TEACHING_GUIDE.md) for presenter prompts, pacing and participant exercises.

## Use as a GitHub repository

Upload the **contents of this project folder** to your repository, so `README.md` and the notebook are at its root. Do not upload the Conda environment or the ZIP itself as the repository's source tree. The included `.gitignore` excludes temporary output, local training checkpoints, notebook checkpoints and machine-specific reports.

The project contains a Windows/Linux CPU workflow under `.github/workflows/test.yml`. It installs dependencies, runs tests and executes the prepared notebook. The workflow runs after upload; its presence is not a claim that a remote CI job has already passed.

Local checks from the activated environment:

```text
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/check_notebook.py
```

The saved notebook contains figures for GitHub readers. Interactive controls require a running Jupyter kernel. An HTML preview is included for reading without starting Jupyter; it contains static figures.

## Files

| Path | Purpose |
|---|---|
| `IAHR_GNN_HKUST_Tutorial.ipynb` | Main teaching notebook |
| `graph_tools.py`, `model.py` | Physical graph, features, GNN and training |
| `network_config.py` | HKUST material wave speeds and shared solver discretisation |
| `signal_story.py`, `signal_ui.py`, `tutorial_ui.py` | Signal interpretation and interactive figures |
| `tsnet_tools.py`, `run_experiments.py` | Local bursts and repeatable evaluation |
| `networks/` | Original INP and upstream provenance/license |
| `models/` | Portable prepared model |
| `data/` | Source locations, hydraulic signals, model configuration and verification |
| `scripts/` | Device installation, environment checks and notebook execution |
| `tests/` | Graph, model, observations, setup and interface checks |

## Citation and credits

Related studies, including Waqar and Ghidaoui's passive transient localisation work, are listed in [REFERENCES.md](REFERENCES.md). `CITATION.cff` and `references.bib` provide citation metadata. The supplied HKUST geometry, hydraulic input and provenance are retained in `networks/`; third-party software remains under its own license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
