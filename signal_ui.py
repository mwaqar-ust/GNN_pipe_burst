"""Interactive signal-first scenes for the GNN teaching notebook.

Every scene also has a static Figure for saved notebook outputs and export.
Controls change one shared set of recorded signals, never a generated sketch
centred on a theoretical arrival.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from signal_story import (event_record, observe_event, graph_arrivals,
                          arrival_table, localise_observation)
from tutorial_ui import NAVY, TEAL, GOLD, PINK, GREY, draw_network


def _short(name):
    return str(name)


def _sensor_index(net, sensor):
    if isinstance(sensor,str):
        if sensor not in net.sensors:
            raise ValueError('Choose one of the current sensor node IDs.')
        return list(net.sensors).index(sensor)
    s = int(sensor)
    if s != sensor or not 0 <= s < net.S:
        raise ValueError('Choose a sensor index present in this network.')
    return s


def _clean_axis(ax):
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=10)


def _known_event_map(net,dataset,index=0,reveal=True):
    event = event_record(net,dataset,index)
    fig,ax = plt.subplots(figsize=(9,6),layout='constrained')
    source = (event['pipe'],event['xi']) if reveal else None
    draw_network(net,ax,source=source)
    title = 'A known event for checking first-arrival physics' if reveal else 'The source is hidden. These sensors recorded the event.'
    ax.set_title(title,loc='left',fontsize=13)
    ax.legend(loc='upper right',frameon=False,fontsize=10)
    return fig


def _plot_record(net,dataset,index=0,sensor=0,response=False,all_sensors=False):
    event = event_record(net,dataset,index)
    s = _sensor_index(net,sensor)
    fig,ax = plt.subplots(figsize=(12,4.8),layout='constrained')
    values = event['signal'] if response else event['pressure_head']
    chosen = range(net.S) if all_sensors else [s]
    for j in chosen:
        ax.plot(event['time'],values[:,j],lw=1.5 if j==s else 1.,
                label=_short(net.sensors[j]),alpha=1 if j==s else .65)
    ax.set(xlabel='Time in the recorded window (s)',
           ylabel='Burst response (m head)' if response else 'Pressure head (m)',
           title='Response after subtracting the no-burst simulation' if response else 'Pressure recorded at the sensor')
    ax.legend(frameon=False,ncol=min(5,len(list(chosen))),fontsize=10)
    _clean_axis(ax)
    return fig


def plot_mystery(net,dataset,index=0,sensor=0,reveal=False):
    """Opening scene: hidden event source, pressure record and burst response."""
    event = event_record(net,dataset,index)
    s = _sensor_index(net,sensor)
    fig = plt.figure(figsize=(13.5,6.5),layout='constrained')
    grid = fig.add_gridspec(2,2,width_ratios=[1.05,1.25])
    map_ax = fig.add_subplot(grid[:,0])
    draw_network(net,map_ax,source=(event['pipe'],event['xi']) if reveal else None,
                 highlight_nodes=[net.sensor_indices[s]])
    map_ax.set_title('Where did the disturbance originate?',loc='left',fontsize=13)
    map_ax.legend(frameon=False,fontsize=9,loc='upper right')
    raw_ax = fig.add_subplot(grid[0,1])
    raw_ax.plot(event['time'],event['pressure_head'][:,s],color=NAVY,lw=1.4)
    raw_ax.set(ylabel='Pressure head (m)',title=f'Recorded pressure at {_short(net.sensors[s])}')
    response_ax = fig.add_subplot(grid[1,1],sharex=raw_ax)
    response_ax.plot(event['time'],event['signal'][:,s],color=TEAL,lw=1.4)
    response_ax.set(xlabel='Time in recorded window (s)',ylabel='Response (m head)',
                    title='Burst response after subtracting a no-burst simulation')
    for ax in [raw_ax,response_ax]:_clean_axis(ax)
    fig.suptitle(f'Passive transient monitoring | event {int(index)+1:02d}',fontsize=16,fontweight='bold',color=NAVY)
    return fig


def plot_arrival_overlay(net,dataset,index=0,sensor=0):
    """Show a graph prediction and an independently detected signal arrival."""
    event = event_record(net,dataset,index)
    obs = observe_event(net,dataset,index)
    expected = graph_arrivals(net,dataset,index)
    s = _sensor_index(net,sensor)
    fig,axes = plt.subplots(1,2,figsize=(13,4.7),layout='constrained',
                           gridspec_kw={'width_ratios':[1.35,1]})
    a,b = axes
    a.plot(event['time'],event['signal'][:,s],color=NAVY,lw=1.5,label='TSNet burst response')
    a.axvline(expected[s],color=GOLD,lw=2,ls='--',label='Graph: onset + travel time')
    if np.isfinite(obs['signal_picks_s'][s]):
        a.axvline(obs['signal_picks_s'][s],color=TEAL,lw=1.7,ls=':',label='Signal-only arrival pick')
    start = max(0,expected[s]-.12)
    a.set_xlim(start,min(event['time'][-1],expected[s]+.45))
    a.set(xlabel='Time in recorded window (s)',ylabel='Response (m head)',
          title=f'First arrival at {_short(net.sensors[s])}')
    a.legend(frameon=False,fontsize=9,loc='best')
    errors = 1000*(obs['signal_picks_s']-expected)
    b.axhline(0,color=GREY,lw=1)
    b.bar(np.arange(net.S),errors,color=[GOLD if j==s else TEAL for j in range(net.S)])
    b.set_xticks(np.arange(net.S),[_short(n) for n in net.sensors],rotation=45,ha='right')
    b.set(ylabel='Signal pick minus graph prediction (ms)',title='First-arrival differences across sensors')
    for ax in axes:_clean_axis(ax)
    return fig


def _plot_arrival_scatter(net,dataset,index=0):
    table = arrival_table(net,dataset,index)
    fig,ax = plt.subplots(figsize=(11.5,4.5),layout='constrained')
    x = np.arange(net.S)
    ax.plot(x,table['Graph arrival (s)'],'o-',color=GOLD,label='Graph prediction')
    ax.plot(x,table['Signal pick (s)'],'x',ms=8,mew=2,color=TEAL,label='Signal-only pick')
    ax.set_xticks(x,[_short(n) for n in net.sensors],rotation=35,ha='right')
    ax.set(ylabel='Arrival time (s)',title='One source produces a timing pattern across sensors')
    ax.legend(frameon=False);_clean_axis(ax)
    return fig


def _plot_noisy_trace(net,dataset,index=0,sensor=0,noise_std_m=.005,jitter_ms=0.,seed=42):
    obs = observe_event(net,dataset,index,noise_std_m,jitter_ms,seed)
    s = _sensor_index(net,sensor)
    fig,axes = plt.subplots(2,1,figsize=(12,6),layout='constrained',height_ratios=[1.3,1])
    a,b = axes
    for ax in axes:
        ax.plot(obs['time'],obs['clean_signal'][:,s],color=GREY,lw=2,label='Clean simulated response')
        ax.plot(obs['time'],obs['noisy_signal'][:,s],color=NAVY,lw=1,label='Response + pressure noise')
        _clean_axis(ax)
    a.set(ylabel='Response (m head)',title=f'{_short(net.sensors[s])}: noise is added to the recorded samples')
    a.legend(frameon=False,fontsize=9)
    picked = obs['signal_picks_s'][s]
    center = obs['clean_picks_s'][s]
    if not np.isfinite(center):center=obs['time'][len(obs['time'])//4]
    if np.isfinite(picked):
        b.axvline(picked,color=TEAL,ls=':',lw=2,label='Repicked arrival')
        b.axvline(obs['arrivals_s'][s],color=PINK,ls='--',lw=1.6,label='After extra pick error + rounding')
    else:
        b.text(.03,.92,'No sustained arrival detected',transform=b.transAxes,va='top',color=PINK)
    threshold = obs['details']['threshold_m'][s]
    b.axhspan(-threshold,threshold,color=TEAL,alpha=.09,label='Detector threshold band')
    b.set_xlim(max(0,center-.15),min(obs['time'][-1],max(center,picked if np.isfinite(picked) else center)+.35))
    within = (obs['time']>=b.get_xlim()[0])&(obs['time']<=b.get_xlim()[1])
    local_values = obs['noisy_signal'][within,s]
    pad = max(3*noise_std_m,.05*(np.ptp(local_values)+1e-8),1e-4)
    b.set_ylim(min(local_values.min(),-threshold)-pad,max(local_values.max(),threshold)+pad)
    b.set(xlabel='Time in recorded window (s)',ylabel='Response (m head)',title='Zoom: waveform noise and additional timestamp error')
    b.legend(frameon=False,fontsize=8,loc='best')
    return fig


def _plot_noisy_picks(net,dataset,index=0,noise_std_m=.005,jitter_ms=0.,seed=42):
    obs = observe_event(net,dataset,index,noise_std_m,jitter_ms,seed)
    fig,ax = plt.subplots(figsize=(12,4.5),layout='constrained')
    x = np.arange(net.S)
    induced = 1000*(obs['signal_picks_s']-obs['clean_picks_s'])
    final = 1000*(obs['arrivals_s']-obs['clean_picks_s'])
    ax.axhline(0,color=GREY,lw=1)
    ax.bar(x-.18,induced,width=.36,color=TEAL,label='Change caused by waveform noise')
    ax.bar(x+.18,final,width=.36,color=PINK,label='After extra pick error and sampling-grid rounding')
    for j,detected in enumerate(obs['detected']):
        if not detected:ax.text(j,0,'missing',rotation=90,ha='center',va='bottom',fontsize=9,color=PINK)
    ax.set_xticks(x,[_short(n) for n in net.sensors],rotation=35,ha='right')
    ax.set(ylabel='Shift relative to the clean signal pick (ms)',
           title='Independent noise draws produce different errors at each sensor')
    ax.legend(frameon=False,fontsize=9);_clean_axis(ax)
    return fig


def _plot_noisy_estimate(net,dataset,model,index=0,noise_std_m=.005,jitter_ms=0.,seed=42):
    from graph_tools import network_error
    event = event_record(net,dataset,index)
    obs = observe_event(net,dataset,index,noise_std_m,jitter_ms,seed)
    estimate = localise_observation(net,model,obs)
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    a,b = axes
    predictions = {'GNN estimate':(estimate['pipe'],estimate['xi'],PINK,'o')} if estimate['available'] else None
    draw_network(net,a,source=(event['pipe'],event['xi']),predictions=predictions)
    a.legend(frameon=False,fontsize=9,loc='upper right')
    if not estimate['available']:
        b.axis('off')
        b.text(.03,.85,'Localisation is unavailable',fontweight='bold',fontsize=14,color=PINK,transform=b.transAxes)
        b.text(.03,.70,estimate['message']+'\n\nThis model requires an arrival at every selected sensor.\nTry a smaller noise level or inspect the signal.',
               fontsize=11,va='top',wrap=True,transform=b.transAxes)
        return fig
    eligible = np.asarray(net.source_pipe_indices)
    probabilities = estimate['probabilities']
    top = eligible[np.argsort(probabilities[eligible])[-5:][::-1]]
    b.barh(np.arange(len(top)),probabilities[top],color=[GOLD if p==event['pipe'] else TEAL for p in top])
    b.set_yticks(np.arange(len(top)),[_short(net.pipe_names[p]) for p in top])
    b.invert_yaxis();b.set_xlim(0,1)
    error = float(np.asarray(network_error(net,event['pipe'],event['xi'],estimate['pipe'],estimate['xi'])).reshape(-1)[0])
    a.set_title(f'GNN: {_short(net.pipe_names[estimate["pipe"]])}, {estimate["xi"]:.3f}L',fontsize=12,loc='left')
    b.set(title=f'Network-distance error: {error:.1f} m',xlabel='Pipe probability (not a calibrated confidence interval)')
    _clean_axis(b)
    return fig


def plot_uncertainty(net,dataset,index=0,sensor=0,noise_std_m=.005,jitter_ms=0.,seed=42,model=None):
    """Static uncertainty scene. The GNN is available in the interactive tabs."""
    return _plot_noisy_trace(net,dataset,index,sensor,noise_std_m,jitter_ms,seed)


def plot_localisation(net,dataset,model,index=0,noise_std_m=.005,jitter_ms=10.,seed=42):
    """Reveal the source beside the GNN estimate and its pipe probabilities."""
    return _plot_noisy_estimate(net,dataset,model,index,noise_std_m,jitter_ms,seed)


def _event_controls(net,dataset):
    import ipywidgets as w
    event = w.Dropdown(options=[(f'Event {i+1:02d}',i) for i in range(len(dataset['pipe']))],
                       value=0,description='Event:',layout=w.Layout(width='210px'))
    sensor = w.Dropdown(options=[(_short(n),i) for i,n in enumerate(net.sensors)],
                        value=0,description='Sensor:',layout=w.Layout(width='210px'))
    return event,sensor


def _tabbed_controls(controls,titles,renderers,explanation):
    """Render only the visible tab, and keep callbacks inspectable for QA."""
    import ipywidgets as w
    from IPython.display import display,clear_output,HTML
    outputs = [w.Output() for _ in titles]
    tabs = w.Tab(children=outputs)
    for i,title in enumerate(titles):tabs.set_title(i,title)
    rows = []
    for offset in range(0,len(controls),3):
        rows.append(w.HBox(controls[offset:offset+3],layout=w.Layout(flex_flow='row wrap')))
    box = w.VBox([w.HTML(explanation),*rows,tabs])
    box._last_error = None
    def render(change=None):
        active = tabs.selected_index
        if active is None:return
        with outputs[active]:
            clear_output(wait=True)
            try:
                item = renderers[active]()
                if isinstance(item,plt.Figure):
                    display(item)
                    plt.close(item)
                else:display(item)
                box._last_error = None
            except Exception as error:
                import html
                box._last_error = str(error)
                display(HTML('<p style="color:#9b3057">'+html.escape(str(error))+'</p>'))
    for control in controls:control.observe(render,names='value')
    tabs.observe(render,names='selected_index')
    box._render = render
    box._tabs = tabs
    box._controls = controls
    render()
    return box


def mystery_explorer(net,dataset):
    """Event/sensor selection with hidden source and three explanatory views."""
    import ipywidgets as w
    event,sensor = _event_controls(net,dataset)
    reveal = w.Checkbox(value=False,description='Reveal simulated source',indent=False,
                        layout=w.Layout(width='240px'))
    all_sensors = w.Checkbox(value=False,description='Show all sensors',indent=False,
                             layout=w.Layout(width='210px'))
    return _tabbed_controls([event,sensor,reveal,all_sensors],
        ['Network','Pressure record','Burst response'],[
            lambda:_known_event_map(net,dataset,event.value,reveal.value),
            lambda:_plot_record(net,dataset,event.value,sensor.value,False,all_sensors.value),
            lambda:_plot_record(net,dataset,event.value,sensor.value,True,all_sensors.value)],
        '<b>Start with the observation.</b> The event occurs without a diagnostic excitation. '
        'Reveal the simulated source only when ready to discuss its location. '
        'The response view subtracts a no-burst TSNet run to remove numerical start-up drift; '
        'it is a controlled simulation example.')


def arrival_explorer(net,dataset):
    """Graph predictions beside independently detected arrivals and timing data."""
    event,sensor = _event_controls(net,dataset)
    return _tabbed_controls([event,sensor],
        ['Signal overlay','Across sensors','Timing values','Known simulation'],[
            lambda:plot_arrival_overlay(net,dataset,event.value,sensor.value),
            lambda:_plot_arrival_scatter(net,dataset,event.value),
            lambda:arrival_table(net,dataset,event.value).round(4),
            lambda:_known_event_map(net,dataset,event.value,True)],
        '<b>Check the first-arrival prior.</b> Dashed lines use the known simulated onset '
        'plus graph travel time for this illustration. Signal picks use only the recordings. '
        'The localisation model receives relative arrival times, not that onset or the source. '
        'Small differences remain because the graph describes earliest travel, while TSNet '
        'also includes discretisation, boundary interactions and signal shape.')


def uncertainty_explorer(net,dataset,model=None,*,noise_std_m=.005,jitter_ms=0.,seed=42):
    """Pressure noise and extra timing uncertainty have separate controls."""
    import ipywidgets as w
    event,sensor = _event_controls(net,dataset)
    noise_std_m,jitter_ms = float(noise_std_m),float(jitter_ms)
    if (not np.isfinite([noise_std_m,jitter_ms]).all()
            or not 0 <= noise_std_m <= .10 or not 0 <= jitter_ms <= 100
            or int(seed) != seed or not 0 <= seed <= 999):
        raise ValueError('Explorer presets require noise between 0 and 0.10 m, extra jitter between 0 and 100 ms, and an integer seed from 0 to 999.')
    noise = w.FloatSlider(value=noise_std_m,min=0,max=.10,step=.001,readout_format='.3f',
                           description='Noise σ (m):',continuous_update=False,
                           style={'description_width':'100px'},layout=w.Layout(width='350px'))
    jitter = w.FloatSlider(value=jitter_ms,min=0,max=100,step=1,readout_format='.0f',
                            description='Extra σ (ms):',continuous_update=False,
                            style={'description_width':'100px'},layout=w.Layout(width='350px'))
    seed = w.IntSlider(value=int(seed),min=0,max=999,description='Noise seed:',continuous_update=False,
                        style={'description_width':'100px'},layout=w.Layout(width='350px'))
    args = lambda:dict(index=event.value,noise_std_m=noise.value,jitter_ms=jitter.value,seed=seed.value)
    renderers = [lambda:_plot_noisy_trace(net,dataset,sensor=sensor.value,**args()),
                 lambda:_plot_noisy_picks(net,dataset,**args()),
                 lambda:arrival_table(net,dataset,**args()).round(4)]
    titles = ['Noisy signal','Arrival shifts','Timing values']
    if model is not None:
        renderers.append(lambda:_plot_noisy_estimate(net,dataset,model,**args()))
        titles.append('GNN estimate')
    return _tabbed_controls([event,sensor,noise,jitter,seed],titles,renderers,
        '<b>Two distinct uncertainty sources.</b> Noise σ changes pressure-head samples '
        'and the detector is run again. Extra σ adds optional independent timing error '
        'after picking. Both are standard deviations, not fixed shifts. The seed keeps '
        'draws reproducible, and each sensor gets its own draws. Final timestamps are '
        'rounded to the 100 Hz grid. Missing detections stay missing; this GNN then '
        'reports that a complete observation is unavailable.')
