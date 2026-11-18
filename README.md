# Graph Neural Networks for Passive Transient Source Localisation

**Can the arrival times of a pressure wave reveal where a pipe burst occurred?** This tutorial uses a graph neural network (GNN) to identify the source pipe and estimate the burst position along it from pressure records at a small number of sensors.

The worked example uses a simplified model of the Hong Kong University of Science and Technology (HKUST) freshwater network. I developed this notebook for the IAHR Training Course on AI Tools for Transient Flow Analyses.

[Watch my live demonstration of the notebook](https://www.iahr.org/video/clip?id=1988).

[Open the tutorial notebook](IAHR_GNN_HKUST_Tutorial.ipynb) to read the worked example and view the saved figures on GitHub. To use the interactive plots, change the sensor layout or simulate a burst, run the notebook locally.

## Run the notebook

You need Conda (Anaconda or Miniconda) and an internet connection for the initial installation. A CPU is sufficient.

1. Clone the repository and enter its folder. You can also choose **Code → Download ZIP**, extract it and open a terminal in the folder containing `environment.yml`.

   ```sh
   git clone https://github.com/mwaqar-ust/GNN_pipe_burst.git
   cd GNN_pipe_burst
   ```

2. Run the setup command for your system. It creates the Python 3.12 environment and registers the Jupyter kernel.

   | System | Command | CPU-only installation |
   | --- | --- | --- |
   | Windows (Anaconda or Miniconda Prompt) | `setup_windows.bat` | `setup_windows.bat --cpu` |
   | Linux (terminal with Conda) | `bash setup_unix.sh` | `bash setup_unix.sh --cpu` |

3. Launch the notebook from the same folder.

   ```sh
   conda activate iahr-gnn-hkust-tutorial
   jupyter lab IAHR_GNN_HKUST_Tutorial.ipynb
   ```

4. Select **Python (IAHR GNN HKUST Tutorial)**, then **Kernel → Restart Kernel and Run All Cells**. Leave `RETRAIN = False` for the first run. This loads the supplied model and simulated pressure records.

## How localisation works

The tutorial represents the network as a graph of **31 nodes and 33 pipes**, with pressure sensors at **10 junctions**. By default, bursts can occur anywhere inside the 21 pipes at least 100 m long. All 33 pipes remain in the wave propagation model.

1. **Measure relative arrivals.** Identify the first arrival in each sensor’s pressure record, then subtract the earliest arrival across sensors. The resulting delays can be calculated even when the burst onset time is unknown.
2. **Learn from graph travel times.** Pipe lengths and wave speeds determine the expected travel times from a candidate source to the sensors. These calculations generate labelled training examples. The GNN combines the timing features with information from neighbouring nodes to predict a pipe and a position measured as a fraction of its length from its start node.
3. **Test on simulated bursts.** TSNet independently simulates pressure waves for held-out source locations. The notebook picks arrivals from the resulting 100 Hz records and compares each predicted location with the known burst location.

## Results from the supplied model

The clean and noisy cases below use the **same 21 simulated bursts**, one per eligible pipe. The noisy case adds pressure-head noise (0.005 m standard deviation) and arrival-time jitter (10 ms standard deviation).

| TSNet pressure records | Correct pipe predictions | Mean network-distance error |
| --- | ---: | ---: |
| Clean | 21/21 (100%) | 21.2 m |
| With pressure noise and timing jitter | 19/21 (90.5%) | 32.2 m |

Distance error is the shortest distance along the pipes between the true and predicted positions. It includes predictions on the wrong pipe. Even with every pipe correctly identified in the clean case, the estimated position along the pipe was off by 21.2 m on average. [Full evaluation results](data/metrics.json)

## Try a different sensor layout

In the first code cell, set `SENSOR_NODES` to a list of junction IDs, such as `['J10', 'J84', 'J102']`, and set `RETRAIN = True`. Restart the kernel and run all cells. The new model is saved to `outputs/gnn_user.npz`. To isolate the effect of sensor placement, keep the number of message-passing layers fixed across runs. The supplied model has two layers, and the notebook’s retraining call defaults to three.

## Data, citation and licence

- [Network geometry and hydraulic assumptions](networks/provenance.json) and [EPANET input file](networks/HKUST.inp)
- [Simulated burst records](data/tsnet_events/) and [pretrained GNN](models/gnn_demo.npz)
- [Citation for this tutorial](CITATION.cff) and [related publications](REFERENCES.md)
- [Repository licence](LICENSE) and [notice accompanying the network files](networks/SOURCE_LICENSE.md)
