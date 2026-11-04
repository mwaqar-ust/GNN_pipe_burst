"""Rebuild the teaching notebook's source cells. Run from any directory."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md(r'''
<div style="background:#15324f;color:white;padding:26px 30px;border-radius:12px">
<div style="color:#89e0df;letter-spacing:2px;font-size:12px">IAHR · A ONE-HOUR LEARNING TUTORIAL</div>
<h1 style="color:white;margin:12px 0">Graph Neural Networks<br>for Passive Transient Source Localisation</h1>
<p style="font-size:18px">HKUST case network · From a recorded pressure disturbance to a location along a pipe.</p>
<p style="color:#d4e4ef">Muhammad Waqar · Hong Kong Metropolitan University</p>
</div>

**The question:** pressure sensors recorded a disturbance. Which pipe did it start in, and where along that pipe?

By the end, you will connect a physical pipe network to a graph, interpret node features and embeddings, follow message passing, and build a GNN with two outputs. The worked example uses the retained HKUST primary network throughout.

**How to use this notebook:** run the setup once, then follow the story. Each topic has a visible figure and an **Explore interactively** panel with sub-tabs. Panel controls apply to that panel; the main walkthrough follows prepared event 01. Prepared weights and hydraulic signals are included. The default **Run All** completes without retraining or running new burst simulations.

| Minutes | Story |
|---|---|
| 0–8 | The signal, the engineering question and brief context |
| 8–24 | Graphs, adjacency and physical arrival times |
| 24–31 | Uncertain signals and uncertain picks |
| 31–44 | Features, embeddings, message passing and two outputs |
| 44–50 | Creating examples and training the GNN |
| 50–60 | Locate the source, interpret errors and explore sensors |
''')
md(r'''
## 0. Prepare the experiment

Select the **Python (IAHR GNN HKUST Tutorial)** kernel. **Shift + Enter** runs one cell. **Kernel → Restart Kernel and Run All Cells** starts a complete run. See [README.md](README.md) for the fresh Conda installation.

The controls below are the main settings. For the prepared walkthrough, leave `RETRAIN=False`. To fit a new GNN, set it to `True` and choose the maximum number of epochs. The setup then trains once and the rest of this notebook uses that newly trained model. Its files go into `outputs/`, preserving the supplied model.

`DEVICE='auto'` uses a working CUDA GPU when available, otherwise CPU. Model and data checks prevent a saved model from being applied to a different sensor order or physical graph.

The default ten sensors use their original EPANET names. A different sensor list or source-pipe threshold requires `RETRAIN=True`. Changing the split seed or counts also requires new hydraulic test sources; keep those fixed for the prepared walkthrough.
''')
code('''
from pathlib import Path
import time, warnings, sys
warnings.filterwarnings('ignore', message='pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='Not all curves were used.*')
warnings.filterwarnings('ignore', message='Changing the headloss formula from H-W to D-W.*')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown
from matplotlib_inline.backend_inline import set_matplotlib_formats
from graph_tools import (load_network, make_source_split, travel_times,
    relative_times, add_timing_noise, network_error, ambiguity_report)
from model import GraphSourceGNN, train_model, load_model, save_model, predict
from run_experiments import load_test_set, evaluate
import tutorial_ui as ui
import signal_ui as signals
from signal_story import observe_event, arrival_table

STARTED = time.perf_counter()
ROOT = Path.cwd()
assert (ROOT / 'networks/HKUST.inp').exists(), 'Open this notebook inside the extracted project folder.'
# Main controls. Change them here, then restart the kernel and run all cells.
RETRAIN = False
TRAINING_EPOCHS = 500
DEVICE = 'auto'
SENSOR_NODES = None  # Example: ['J10', 'J84', 'J102']
MIN_SOURCE_PIPE_LENGTH = 100.0  # metres; all pipes still carry waves in the graph
SAMPLING_HZ = 100.0
TIMING_STD_S = 0.010            # independent Gaussian timestamp uncertainty
TIME_QUANTUM_S = 1.0 / SAMPLING_HZ
BACKGROUND_STD_M = 0.005       # head-noise standard deviation for the signal demonstration
EXTRA_PICK_JITTER_MS = 10.0    # separate from the error caused by waveform noise
SHOW_NODE_IDS = False  # enable for the full map; focused neighbourhoods show IDs
SHOW_PIPE_IDS = False

# Small graphs benefit from limiting CPU thread overhead.
torch.set_num_threads(2)
ui.style()
set_matplotlib_formats('png')
net = load_network(sensors=SENSOR_NODES, min_source_pipe_length=MIN_SOURCE_PIPE_LENGTH)
splits = make_source_split(net, train_per_pipe=48, val_per_pipe=12, test_per_pipe=2, seed=2026)
tsdata = load_test_set(net, splits=splits)
if RETRAIN:
    gnn, history = train_model(net, splits['train'], splits['val'],
        epochs=TRAINING_EPOCHS, sigma=TIMING_STD_S, quantum=TIME_QUANTUM_S,
        seed=42, hidden=32, layers=2, batch_size=128, learning_rate=0.002,
        position_weight=4.0, patience=30, device=DEVICE, threads=2,
        checkpoint_path=ROOT/'outputs/training_last.pt', resume=False, print_every=10)
    save_model(gnn, ROOT/'outputs/gnn_user.npz', history=history)
else:
    if TIMING_STD_S != 0.010 or TIME_QUANTUM_S != 0.010:
        raise ValueError('The supplied model was trained with 10 ms Gaussian jitter and 100 Hz timestamps. Set RETRAIN=True for a new training noise setting.')
    gnn = load_model(net, ROOT/'models/gnn_demo.npz', device=DEVICE)
    history = gnn.history
print(f'Ready | {next(gnn.parameters()).device} | {net.N} nodes | {net.P} physical pipes | {net.S} sensors')
print(f'Eligible source pipes: {net.source_pipe_count} (length >= {MIN_SOURCE_PIPE_LENGTH:g} m)')
print('Sensor order:', ', '.join(net.sensors))
print('Training / validation / held-out graph locations:', *(len(splits[s]['pipe']) for s in ('train','val','test')))
print('Active model:', 'newly trained model in outputs/' if RETRAIN else 'prepared teaching model')
print(f'Setup completed in {time.perf_counter()-STARTED:.1f} seconds.')
''')
md(r'''
## 1. We have a pressure signal. Where did it come from?

A burst releases water and launches pressure disturbances. Waves travel through the connected pipes and reach the sensors at different times. **Passive monitoring** observes these events without imposing a diagnostic excitation.

Our sensors sit at existing junctions. The burst may start **inside any eligible pipe**. Its location and onset time are unknown to the localisation model.

The first display hides the simulated source. Inspect the waveforms and identify what information a sensor actually provides. The pressure-head axis is in metres of water; this is a pressure measurement expressed as head.

**Discuss:** does one pressure trace determine a unique location? What extra information do several sensors provide?
''')
code('''
fig = signals.plot_mystery(net, tsdata, index=0, sensor=0, reveal=False)
display(fig); plt.close(fig)
display(ui.panel(signals.mystery_explorer(net, tsdata), 'Explore interactively: the unknown event'))
''')
md(r'''
### The physical network

The retained HKUST primary network contains **30 junctions and the fixed-head boundary T1**, connected by **33 pipes**. The GNN retains all 31 nodes. T1 is represented as a 40 m head reservoir in this controlled hydraulic example. Geographic coordinates are projected locally for drawing; travel times and distance errors use the measured pipe lengths.

For this lesson, potential source pipes are at least **100 m** long: 21 pipes meet that threshold. Shorter pipes remain in the network and in wave propagation. The threshold controls where a source may be placed, not which pipes physically exist.

Use the interactive label switches or the focused-neighbourhood tab to read individual IDs. Labels such as **J10**, **T1** and **P110_2** retain their original identifiers. They are not Python array positions.
''')
code('''
fig = ui.plot_network(net, show_node_ids=SHOW_NODE_IDS, show_pipe_ids=SHOW_PIPE_IDS)
display(fig); plt.close(fig)
display(ui.panel(ui.network_explorer(net), 'Explore interactively: network and labels'))
''')
md(r'''
### Where this lesson fits

Transient source localisation connects wave propagation, sensor information and inverse estimation. Waqar and Ghidaoui's [ICHD conference contribution](https://researchportal.hkust.edu.hk/en/publications/passive-transient-source-localization-in-water-distribution-netwo/) provides related application context. [Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907) introduce graph convolution, while [Xing and Sela (2020)](https://doi.org/10.1016/j.advengsoft.2020.102884) describe TSNet, the hydraulic simulator used here.

**Today we will build understanding of GNNs through a worked engineering example.** The model, explanations and exercises form a teaching design. The goal is to understand the ingredients and how they connect.

Our prediction has two parts: **which original pipe** and **the position within that pipe**. We return to the hidden source after constructing those ingredients. Full references are in [REFERENCES.md](REFERENCES.md).
''')
md(r'''
## 2. What is a graph?

Write a graph as $G=(V,E)$: $V$ is its set of nodes and $E$ its set of edges. A weighted graph additionally assigns a value $w_e$ to each edge.

| Concept | Meaning in this tutorial |
|---|---|
| Node | A junction or boundary connection |
| Edge | A physical connection between two nodes |
| Neighbours | Nodes directly connected to the chosen node |
| Degree | Number of neighbouring nodes |
| Path | A sequence of connected edges |
| Hop | Crossing one edge, regardless of its physical length |
| Weight | A specified property, such as length or travel time |

We use an undirected graph because pressure disturbances travel in both directions, neglecting mean-flow advection. This retained network contains pipe connections and a fixed-head supply boundary.

### Store connectivity in a matrix

$$A_{ij}=\begin{cases}1,&\text{an open connection joins }i\text{ and }j,\\0,&\text{otherwise.}\end{cases}$$

The full matrix is $31\times31$. Start with a small neighbourhood taken from **this same network**. Follow one row of the matrix back to the connected nodes. The initial diagonal is zero; self-connections enter the GNN later.
''')
code('''
fig = ui.plot_adjacency(net, node_name='J84', hops=1)
display(fig); plt.close(fig)
display(ui.panel(ui.adjacency_explorer(net), 'Explore interactively: graph ↔ adjacency'))
''')
md(r'''
**Try and explain:** increase the neighbourhood from one hop to two. Which new nodes enter the view? Does one hop represent the same distance everywhere?

Keep three objects distinct: **adjacency** stores connectivity; **physical edge costs** describe travel delays; **learned embeddings** represent information used by the GNN.
''')
md(r'''
## 3. Give the graph a physical meaning

A wave crossing pipe $e$ of length $L_e$ at speed $a_e$ takes

$$w_e=\frac{L_e}{a_e}\quad\text{seconds}.$$

The path with the smallest sum of these costs gives the earliest predicted travel time. For a source at fraction $\xi$ from pipe endpoint $u$ towards endpoint $v$, waves can leave in either direction:

$$T_s(e,\xi)=\min\left[\frac{\xi L_e}{a_e}+D(u,s),\;\frac{(1-\xi)L_e}{a_e}+D(v,s)\right].$$

Here $D(u,s)$ is the shortest travel time from endpoint $u$ to sensor $s$. A source halfway along the pipe has $\xi=0.5$.

**What the prior describes:** first-arrival timing. The full waveform also contains amplitudes, reflections and interactions at boundaries. Moving the source changes its timing signature even though the network topology stays fixed.
''')
code('''
pipe = int(tsdata['pipe'][0])
xi = float(tsdata['xi'][0])  # reveal the same event used in the opening scene
clean_times = travel_times(net, [pipe], [xi])
observation = relative_times(add_timing_noise(clean_times, TIMING_STD_S,
    TIME_QUANTUM_S, np.random.default_rng(12)))
fig = ui.plot_travel(net, pipe=pipe, xi=xi, sigma_ms=1000*TIMING_STD_S)
display(fig); plt.close(fig)
display(ui.panel(ui.source_explorer(net), 'Explore interactively: source position and travel times'))
''')
md(r'''
### Check the prediction against a transient signal

Now reveal the source for a known simulated event. Overlay the graph prediction and the signal-only arrival pick. This is a check of the travel-time approximation, not a manufactured waveform centred on that prediction.

For this explanatory overlay only, the event onset $t_0$ is known from the simulation, so the vertical line is $t_0+T_s$. **The localisation model never receives that onset or the source label.** We will remove the unknown onset from its inputs.

The stored pressure responses are sampled at **100 Hz**, so one sample spans **10 ms**. Differences between the vertical lines may reflect sampling, the detector's threshold crossing, and differences between the graph approximation and the hydraulic simulation.
''')
code('''
fig = signals.plot_arrival_overlay(net, tsdata, index=0, sensor=0)
display(fig); plt.close(fig)
display(ui.panel(signals.arrival_explorer(net, tsdata), 'Explore interactively: predicted and picked arrivals'))
''')
md(r'''
<details><summary><b>Physical and numerical details</b></summary>

T1 is a fixed-head boundary in this TSNet setup. The travel-time graph excludes paths that transmit through it. The controlled hydraulic model starts from zero demands and zero junction elevations, with a 40 m supply head and a fixed Darcy friction factor of 0.02.

Nominal wave speeds use a ductile-iron K9 wall model with each pipe diameter, giving approximately 1,137–1,231 m/s. TSNet adjusts these speeds during discretisation. The prior uses effective speeds computed once on the original **unsplit** network with a requested 1.04045 ms step, before any source is chosen. Event-specific split geometry and adjusted wave speeds are not input features. Geographic drawing coordinates do not replace measured pipe lengths.

The cached traces come from actual TSNet method-of-characteristics simulations. A matched no-burst simulation removes small initialisation drifts for the controlled arrival-picking experiment. Raw pressure heads remain available in the signal tabs. This subtraction is a numerical teaching convenience; operational signals require a detector validated for their own drift and noise.
</details>
''')
md(r'''
## 4. What changes when the observation is uncertain?

A clean leading edge can be easy to pick. Background pressure noise changes the trace and can change the detected arrival. An analyst or detector may introduce further timestamp uncertainty.

We will explore these separately:

1. **Background noise:** add independent Gaussian head noise, in metres, to each sensor's pressure response and rerun the detector.
2. **Additional picking uncertainty:** perturb the detected timestamps with independent Gaussian errors, in milliseconds, then round them to the sampling grid.

The same random seed keeps a slider comparison repeatable. Setting additional picking uncertainty to zero isolates the effect of waveform noise. A missing detection is shown explicitly; the true source or predicted arrival is never used to fill it in.

**The learning question:** can the GNN learn useful location patterns from many observations with uncertainty? The training examples will include timing perturbations. The GNN in this notebook consumes the resulting timing features, not the raw waveform.
''')
code('''
fig = signals.plot_uncertainty(net, tsdata, index=0, sensor=0,
    noise_std_m=BACKGROUND_STD_M, jitter_ms=0.0, seed=42)
display(fig); plt.close(fig)
display(ui.panel(signals.uncertainty_explorer(net, tsdata, noise_std_m=BACKGROUND_STD_M, jitter_ms=0.0, seed=42), 'Explore interactively: noise → picks → timing features'))
''')
md(r'''
### Remove the unknown start time

An observed timestamp can be written as $t_s=t_0+T_s+\epsilon_s$. Subtracting the earliest observed arrival removes the common, unknown $t_0$:

$$\Delta t_s=t_s-\min_r t_r.$$

Noise is added to the individual timestamps **before** this operation. The individual draws are independent, but the resulting relative-time errors share a reference and therefore need not be independent. The model internally mean-centres the timing vector as well; a common shift contains no location information here.

At 100 Hz, a one-sample error is 10 ms. A signed error of 2–10 samples is 20–100 ms in magnitude, not a standard deviation of 2–10 ms. Our default training uncertainty is **10 ms standard deviation**, with 10 ms timestamp quantisation.
''')
md(r'''
## 5. Features become embeddings

A **feature** is a number we supply. An **embedding** is a vector produced by learned transformations of those features.

A sensor reading at one node would need many local message-passing steps to cross this network. We therefore make the observed sensor timing vector available to every node, alongside known node-to-sensor travel-time landmarks. This is an explicit feature design for a fixed network.

| Input group at each node | Channels with ten sensors | Interpretation |
|---|---:|---|
| Centred node-to-sensor travel times | 10 | Where this node lies in travel-time space |
| Centred observed arrival vector | 10 | What the sensors recorded for this event |
| Observed minus landmark vector | 10 | Signed timing differences |
| Squared differences | 10 | Size of those differences |
| Coordinates, degree, boundary flag, sensor flag, local reading | 6 | Known node properties and local measurement availability |
| **Total** | **46** | Inputs, before learning |

Known graph properties contain no event source label. A learned affine transformation and GELU activation map each 46-component input to a **32-component initial embedding**:

$$H^{(0)}=\operatorname{GELU}(XW_{\mathrm{in}}+b_{\mathrm{in}}).$$

The displayed embeddings use the active trained model loaded or fitted in setup. We inspect its training process later.
''')
code('''
# Use the same recorded event now that its noisy observations have been introduced.
observed_event = observe_event(net, tsdata, index=0, noise_std_m=BACKGROUND_STD_M,
    jitter_ms=EXTRA_PICK_JITTER_MS, seed=42)
observation = observed_event['relative_s'][None, :]
if not np.isfinite(observation).all():
    raise ValueError('This noise setting leaves a missing arrival. Reduce BACKGROUND_STD_M for the feature walkthrough; the interactive signal explorer handles missing picks explicitly.')
fig = ui.plot_features(net, gnn, observation)
display(fig); plt.close(fig)
display(ui.panel(ui.feature_explorer(net, gnn, observation), 'Explore interactively: input features and initial embeddings'))
''')
md(r'''
## 6. Message passing: receive, combine, update

Add self-connections and normalise by node degrees:

$$\widehat A=\widetilde D^{-1/2}(A+I)\widetilde D^{-1/2}.$$

A message-passing layer in this model computes

$$H^{(k+1)}=\operatorname{GELU}\!\left(H^{(k)}W_{\mathrm{self}}^{(k)}+\widehat A H^{(k)}W_{\mathrm{neighbour}}^{(k)}+b^{(k)}\right)+0.2H^{(k)}.$$

Read it in order: **gather neighbour information**, transform it with learned weights, combine it with the node's own information, apply a nonlinear activation, and retain a small residual contribution from the previous representation.

The same learned transformations are applied to every node. Two such layers let each node combine local neighbourhood information over two hops, on top of the sensor timing vector already supplied to every node. These updates are learned computations; they do not solve the water-hammer equations.
''')
code('''
encoded = predict(gnn, observation, return_embeddings=True)
H0 = encoded['embeddings'][0][0]
Ahat = gnn.normalized_adjacency.detach().cpu().numpy()
messages = Ahat @ H0
node = net.names.index('J84')
display(pd.DataFrame({'Before aggregation': H0[node,:6],
                      'Aggregated neighbours + self': messages[node,:6]},
                     index=[f'channel {i+1}' for i in range(6)]).round(4))
fig = ui.plot_embeddings(net, gnn, observation, layer=2)
display(fig); plt.close(fig)
display(ui.panel(ui.embedding_explorer(net, gnn, observation), 'Explore interactively: layers, embeddings and aggregation'))
''')
md(r'''
**Interpret the figure:** layer 0 is the encoded input, while layers 1 and 2 have received graph messages. A heatmap colour describes a learned channel value. The two-dimensional embedding scatter uses PCA for display; its axes are not map coordinates. Individual channels need not correspond to recognisable hydraulic quantities.

**Try:** follow one node through all three layers. Then select a different source event in the source explorer and examine how its predictions change while the graph itself stays fixed.
''')
md(r'''
## 7. Two outputs describe one physical location

For each original pipe, the decoder combines the embeddings at its two endpoints, their absolute difference, observed timing information, endpoint landmarks, and pipe length and diameter. The decoder's weights are shared across pipes.

- The **classification output** assigns one score per pipe. Softmax converts eligible-pipe scores into probabilities.
- The **regression output** supplies a fraction $\widehat\xi_e\in(0,1)$ for each pipe using a sigmoid.

Choose $\widehat e=\arg\max_e p_e$, then use **that pipe's** fraction $\widehat\xi_{\widehat e}$. The distance from its original start node is $\widehat x=\widehat\xi_{\widehat e}L_{\widehat e}$. The true pipe is not given to the model at inference.

All 33 pipe columns retain their original indexing. Pipes below the source-length threshold have zero output probability but remain in graph propagation. Softmax values are model outputs, not calibrated uncertainty intervals.
''')
code('''
fig = ui.plot_architecture(net, gnn)
display(fig); plt.close(fig)
result = predict(gnn, observation)
pred_pipe, pred_xi = int(result['pipe'][0]), float(result['xi'][0])
print('True teaching source:', net.pipe_names[pipe], f'{xi:.3f}L')
print('GNN prediction:', net.pipe_names[pred_pipe], f'{pred_xi:.3f}L')
print(f'Distance from predicted pipe start: {pred_xi*net.lengths[pred_pipe]:.1f} m')
print('Trainable parameters:', sum(p.numel() for p in gnn.parameters()))
display(ui.panel(ui.source_explorer(net, gnn), 'Explore interactively: timing pattern and GNN outputs'))
''')
md(r'''
## 8. Where do the training examples come from?

Thousands of hydraulic simulations would be costly. The graph travel-time prior generates labelled **timing vectors** quickly, providing a surrogate for the first-arrival part of the physics.

For each example, choose an eligible original pipe and a position within it, compute the sensor travel times, add timestamp uncertainty and quantisation, and remove the unknown common onset. The labels are the original pipe ID and position fraction. This process does not generate full pressure waveforms.

| Split | Locations per eligible pipe | Default count | Role |
|---|---:|---:|---|
| Training | 48 | 1,008 | Fit model parameters |
| Validation | 12 | 252 | Choose the checkpoint |
| Held-out graph prior | 2 | 42 | Evaluate unseen locations |

These source locations are disjoint. Fresh timing noise is drawn for training each epoch. Validation noise stays fixed so checkpoint selection remains comparable. Exact junction positions are excluded because a junction belongs to more than one pipe.
''')
code('''
sample = slice(0, 5)
raw_times = add_timing_noise(splits['train']['times'][sample], TIMING_STD_S,
    TIME_QUANTUM_S, np.random.default_rng(42))
examples = pd.DataFrame(relative_times(raw_times), columns=list(net.sensors))
examples.insert(0, 'Source fraction', splits['train']['xi'][sample])
examples.insert(0, 'Source pipe', [net.pipe_names[int(p)] for p in splits['train']['pipe'][sample]])
display(examples.round(3))
''')
md(r'''
### How the model learns

For each batch, the model predicts the pipe scores and fractions. The loss combines cross-entropy for the pipe label with squared fraction error on the **true pipe during training**:

$$\mathcal L=\operatorname{CE}(p,e_{\mathrm{true}})+4\left(\widehat\xi_{e_{\mathrm{true}}}-\xi_{\mathrm{true}}\right)^2.$$

The coefficient 4 balances the two terms; fraction error is dimensionless, while our reported distance error is in metres. Backpropagation computes gradients, and Adam updates the learned weights. The graph and physical pipe properties remain fixed.

The setup cell contains the actual `train_model(...)` call. Set `RETRAIN=True` there to fit a new model. It uses 32 hidden channels, two message-passing layers, batch size 128 and an initial learning rate of 0.002. Validation controls learning-rate reductions and early stopping. The maximum is 500 epochs; 30 epochs without a sufficient validation-loss improvement stop the run. The best validation-loss checkpoint is restored.

**Read the curves:** a falling training loss indicates fit to the noisy training examples. Validation tells us how that fit behaves at different source locations. Neither curve is the final hydraulic test result.
''')
code('''
fig = ui.plot_training(history)
display(fig); plt.close(fig)
print('Selected validation checkpoint: epoch', history.get('best_epoch', '?'))
print('Completed epochs:', history.get('completed_epochs', len(history.get('epoch', []))))
''')
md(r'''
## 9. Return to the pressure signal and locate its source

The hydraulic test events come from TSNet's method-of-characteristics solver. For each event, WNTR inserts a temporary junction inside an original pipe, TSNet applies a burst there, and pressure histories are recorded at the original junctions. The temporary source node never becomes a GNN input node.

We now extract arrivals from the stored **100 Hz** signals, form the timing features and predict the original pipe and fraction. The signal explorer can reveal the true source after prediction.

The prepared cache contains **21 events**, one in each eligible pipe. For every pipe, the first of its two held-out source positions is selected before model evaluation. All 21 events remain in the report; their predictions do not determine which cases are shown.

The GNN has learned from graph-derived timing examples. Testing it on these hydraulic signals checks how that learned mapping behaves with a more complete wave simulation. Training timestamp jitter approximates timing uncertainty; it does not guarantee robustness to every noise level or picking bias.
''')
code('''
fig = signals.plot_localisation(net, tsdata, gnn, index=0,
    noise_std_m=BACKGROUND_STD_M, jitter_ms=EXTRA_PICK_JITTER_MS, seed=42)
display(fig); plt.close(fig)
display(ui.panel(signals.uncertainty_explorer(net, tsdata, model=gnn, noise_std_m=BACKGROUND_STD_M, jitter_ms=EXTRA_PICK_JITTER_MS, seed=42), 'Explore interactively: signals, picks and predicted source'))
''')
md(r'''
### Interpret the two results

**Pipe accuracy:** the percentage of events assigned to their correct original pipe.

**Mean network-distance error:** the shortest physical route between the true and predicted continuous source positions, averaged over events. Wrong-pipe predictions are included. The direct along-pipe distance is also considered when both points lie on the same pipe.

A wrong pipe near a junction can still give a small distance error. Conversely, selecting the correct pipe does not guarantee a good position estimate. Inspect both numbers and individual examples.

The report separates held-out graph-derived observations, clean hydraulic signal picks, and hydraulic observations with background noise plus extra picking uncertainty. Detection coverage is shown if a sensor fails to produce a usable arrival. An unlocalised event cannot be assigned a fabricated distance error.
''')
code('''
report = evaluate(net, splits, gnn, sigma=TIMING_STD_S, quantum=TIME_QUANTUM_S,
                  pressure_noise_std_m=BACKGROUND_STD_M, extra_jitter_ms=EXTRA_PICK_JITTER_MS)
display(ui.metrics_table(report))
fig = ui.plot_metrics(report)
display(fig); plt.close(fig)
''')
md(r'''
## 10. What can the sensors actually identify?

If every sensor is reached through the same endpoint of a source pipe, moving the source along that pipe adds the same delay to all arrivals. Removing the common onset removes that delay too. The relative timing signature is then constant, so position is unidentifiable from these features.

**Try:** use a constant-signature pipe below, set timing noise to zero and move the source. Then inspect another sensor layout. The sensor explorer changes the diagnostic view; changing the model's sensor inputs requires retraining with the new list in setup.

The prepared ten-sensor layout is retained from the HKUST case. It is a reproducible starting configuration, not a claim of an optimum deployment.
''')
code('''
ambiguity = ambiguity_report(net)
print('Eligible pipes with a constant relative-arrival signature:')
print(', '.join(ambiguity['constant_signature_pipes']) or 'None at the checked tolerance.')
display(ui.panel(ui.sensor_explorer(net), 'Explore interactively: sensors and observability'))
''')
md(r'''
## 11. Optional: run a new burst locally

This lab runs only after you press its simulation button. It uses the locally installed TSNet package, simulates the burst and its matched no-burst reference, and applies the GNN to the picked arrivals.

The position fraction is measured from the original pipe start. The **opening time** controls how quickly the burst develops. The coefficient $C_b$ enters $Q_b=C_b\sqrt{H-z}$ and has units $\mathrm{m^3\,s^{-1}\,m^{-1/2}}$ for positive pressure head.

The demonstration starts the burst at 0.5 s in a 3 s record, but the GNN is not given this onset. The default coefficient corresponds to 0.001 m³/s at 40 m pressure head. Very short split segments require smaller simulation time steps. New simulations can take minutes; use the cached signals during the one-hour session.
''')
code('''
display(ui.panel(ui.burst_lab(net, gnn), 'Optional local TSNet burst laboratory'))
''')
md(r'''
## 12. Continue learning

1. **Arrival uncertainty:** move a source along a pipe and increase the timing noise. Explain which changes remain distinguishable.
2. **Two outputs:** find a wrong-pipe prediction with a small distance error. Explain it on the map.
3. **Representations:** inspect a node's features, initial embedding and layer-2 embedding. Identify which steps are learned.
4. **Sensors:** inspect a poorly observed branch, choose a new sensor layout, retrain, and explain what information changed.
5. **Transient physics:** generate a slower-opening burst. Inspect how its leading edge affects the pick.

For an optional participant exercise, submit the settings, one signal/arrival figure, one embedding figure, and the two localisation metrics with a short interpretation. Results need not match a particular percentage.

| File | What you can learn or change |
|---|---|
| `graph_tools.py` | Network, travel-time prior, source splits and distance error |
| `model.py` | Features, residual GNN, pipe decoder and training loop |
| `signal_story.py` | Signal noise, independent picking and observation construction |
| `tutorial_ui.py`, `signal_ui.py` | Figures and interactive sub-tabs |
| `tsnet_tools.py` | The optional burst simulation and arrival detector |
| `docs/TEACHING_GUIDE.md` | Presenter prompts, pacing and exercises |
| `docs/SCIENTIFIC_NOTES.md` | Assumptions, data provenance and interpretation |
| `docs/SETUP.md`, `docs/GPU_SETUP.md` | Fresh Conda setup, GPU selection and troubleshooting |

This example concerns a fixed network and sensor configuration. A new topology or input representation requires corresponding training and evaluation. The graph prior supports efficient learning, while independent hydraulic signals let us inspect how the learned localisation behaves.

See [REFERENCES.md](REFERENCES.md) for the source studies and software, and [CITATION.cff](CITATION.cff) to cite this tutorial.
''')
code('''
print(f'Notebook complete in {time.perf_counter()-STARTED:.1f} seconds.')
print('Optional simulations run only when their button is pressed.')
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python (IAHR GNN HKUST Tutorial)', 'language': 'python', 'name': 'iahr-gnn-hkust-tutorial'},
               'language_info': {'name': 'python', 'version': '3.12'},
               'title': 'Graph Neural Networks for Passive Transient Source Localisation'}
nbf.write(nb, ROOT/'IAHR_GNN_HKUST_Tutorial.ipynb')
print(f'Built {len(cells)} cells ({sum(c.cell_type=="code" for c in cells)} code cells).')
