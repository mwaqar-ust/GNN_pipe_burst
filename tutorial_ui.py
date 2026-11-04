"""Teaching figures and interactive sub-tabs for the GNN source tutorial.

The plotting functions return Matplotlib figures for static notebook previews.
The explorer functions return live ipywidgets controls, using the same figures.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
NAVY, TEAL, GOLD, PINK, GREY = '#15324f', '#087f8c', '#d89427', '#b14a70', '#b3bec9'
PALE = '#edf4f7'


def style():
    """Readable defaults for a projected notebook and exported PNG figures."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelcolor': NAVY,
        'text.color': NAVY, 'axes.edgecolor': '#d6dee6', 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'figure.dpi': 105})


def _node_label(name):
    return str(name)


def _pipe_label(name):
    return str(name)


def pipe_point(net, pipe, xi):
    u, v = net.ends[int(pipe)]
    return (1-float(xi))*net.positions[u] + float(xi)*net.positions[v]


def draw_network(net, ax=None, source=None, predictions=None, labels=False,
                 highlight_nodes=None, sensor_indices=None, show_pipe_ids=False):
    """Draw the fixed graph; `labels` enables every node ID, with no duplicates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    xy = np.asarray(net.positions)
    ax.add_collection(LineCollection(xy[net.link_ends], colors='#bbc7d1', linewidths=1.1, zorder=1))
    ax.scatter(*xy.T, s=11, color=NAVY, alpha=.75, zorder=2)
    bounds = np.asarray(net.boundary_indices, dtype=int)
    if len(bounds):
        ax.scatter(*xy[bounds].T, s=55, marker='s', color=GREY, edgecolor=NAVY,
                   label='Fixed-head boundary T1', zorder=4)
    sensors = np.asarray(net.sensor_indices if sensor_indices is None else sensor_indices, dtype=int)
    ax.scatter(*xy[sensors].T, s=60, color=TEAL, edgecolor='white', linewidth=1,
               label='Pressure sensors', zorder=5)
    for i in (range(net.N) if labels else sensors):
        ax.annotate(_node_label(net.names[i]), xy[i], xytext=(3, 3), textcoords='offset points',
                    fontsize=6.5 if labels else 8, color=TEAL if i in sensors else NAVY,
                    zorder=10)
    if show_pipe_ids:
        for name, (u,v) in zip(net.pipe_names, net.ends):
            mid = (xy[u]+xy[v])/2
            ax.annotate(_pipe_label(name), mid, xytext=(0,-5), textcoords='offset points',
                        ha='center', va='top', fontsize=5.8, color='#97522c', zorder=9,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.80,'pad':.25})
    if highlight_nodes is not None:
        chosen = np.asarray(highlight_nodes, dtype=int)
        ax.scatter(*xy[chosen].T, s=105, facecolors='none', edgecolors=PINK, zorder=6)
    if source is not None:
        p, x = int(source[0]), float(source[1])
        ax.plot(*xy[net.ends[p]].T, color=GOLD, linewidth=3, zorder=3)
        ax.scatter(*pipe_point(net,p,x), marker='*', s=210, color=GOLD, edgecolor=NAVY,
                   linewidth=.6, label='True source', zorder=7)
    for label, (p,x,color,marker) in (predictions or {}).items():
        ax.scatter(*pipe_point(net,p,x), marker=marker, s=115, facecolors='none',
                   edgecolors=color, linewidths=2, label=label, zorder=8)
    ax.autoscale(); ax.margins(.08); ax.set_aspect('equal'); ax.axis('off')
    return ax


def plot_network(net, show_node_ids=True, show_pipe_ids=True):
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_network(net,ax,labels=show_node_ids,show_pipe_ids=show_pipe_ids)
    ax.add_collection(LineCollection(net.positions[net.ends[net.source_pipe_indices]],
        colors=TEAL, linewidths=1.6, alpha=.65, zorder=1.5,
        label=f'Source pipes: length ≥ {net.min_source_pipe_length:g} m'))
    ax.set_title(f'HKUST | {net.N} nodes, {net.P} physical pipes, {net.S} pressure sensors\n'
                 f'{net.source_pipe_count} eligible source pipes; every physical connection remains', loc='left')
    ax.legend(loc='upper right', bbox_to_anchor=(1,1), frameon=False, fontsize=9)
    fig.text(.02,.015,'J10, T1 and P110_2 are original network IDs, not array indices. '
             'Map coordinates are for drawing; pipe lengths define distance.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    return fig


def _neighbourhood(net,node_name,hops):
    center = net.names.index(node_name)
    reached,frontier = {center},{center}
    for _ in range(int(hops)):
        frontier = {int(j) for i in frontier for j in np.flatnonzero(net.adjacency[i])}-reached
        reached |= frontier
    return center,sorted(reached)


def plot_adjacency(net,node_name='J84',hops=1,matrix='binary'):
    center,ids = _neighbourhood(net,node_name,hops)
    reached = set(ids)
    fig,axes = plt.subplots(1,2,figsize=(12.5,5.3),gridspec_kw={'width_ratios':[1,1.1]})
    for u,v in net.link_ends:
        if u in reached and v in reached:
            axes[0].plot(*net.positions[[u,v]].T,color=GREY,lw=2)
    for i in ids:
        color = GOLD if i==center else TEAL if i in net.sensor_indices else NAVY
        axes[0].scatter(*net.positions[i],s=155,color=color,zorder=3)
        axes[0].annotate(_node_label(net.names[i]),net.positions[i],xytext=(7,7),
                         textcoords='offset points',fontsize=9)
    axes[0].axis('off'); axes[0].margins(.26)
    axes[0].set_title(f'{int(hops)}-hop neighbourhood of {_node_label(node_name)}')
    full = np.asarray(net.adjacency,dtype=float)
    if matrix=='self':
        full = full+np.eye(net.N)
        title = 'Add self-connections: A + I'
    elif matrix=='normalised':
        full = full+np.eye(net.N)
        d = full.sum(axis=1)
        full = full/np.sqrt(d[:,None]*d[None,:])
        title = 'Normalised adjacency: weights for aggregation'
    elif matrix=='binary':
        title = 'Binary adjacency: connected = 1'
    else:
        raise ValueError("matrix must be 'binary', 'self' or 'normalised'.")
    a = full[np.ix_(ids,ids)]
    im=axes[1].imshow(a,cmap='Blues',vmin=0,vmax=1)
    labels=[_node_label(net.names[i]) for i in ids]
    axes[1].set_xticks(range(len(ids)),labels,rotation=60,ha='right',fontsize=8)
    axes[1].set_yticks(range(len(ids)),labels,fontsize=8)
    if len(ids)<=16:
        for i in range(len(ids)):
            for j in range(len(ids)):
                val = f'{a[i,j]:.2f}' if matrix=='normalised' and a[i,j] else str(int(a[i,j]))
                axes[1].text(j,i,val,ha='center',va='center',color='white' if a[i,j]>.5 else NAVY,
                             fontsize=8)
    axes[1].set_title(title)
    fig.text(.02,.015,'A hop is a connection, not a distance. Normalisation uses degrees in the full graph. '
             'Only the selected rows and columns are displayed.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    return fig


def link_table(net,node_name=None,hops=1):
    import pandas as pd
    included = set(range(net.N)) if node_name is None else set(_neighbourhood(net,node_name,hops)[1])
    rows=[]
    for p,(u,v) in enumerate(net.ends):
        if u in included and v in included:
            rows.append({'Pipe':net.pipe_names[p],'Start node':net.names[u],'End node':net.names[v],
                'Length (m)':net.lengths[p],'Wave speed (m/s)':net.pipe_wave_speeds[p],
                'Travel cost L/a (s)':net.lengths[p]/net.pipe_wave_speeds[p],
                'Eligible source':bool(net.source_pipe_mask[p])})
    return pd.DataFrame(rows).round(4)


def plot_travel(net,pipe=None,xi=.4,sigma_ms=10,seed=10,quantum=.010):
    from graph_tools import travel_times,relative_times,add_timing_noise
    pipe = int(net.source_pipe_indices[0]) if pipe is None else int(pipe)
    clean=travel_times(net,[pipe],[xi])
    noisy=add_timing_noise(clean,float(sigma_ms)/1000,quantum,np.random.default_rng(seed))
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={'width_ratios':[1.05,1]})
    draw_network(net,axes[0],source=(pipe,xi))
    axes[0].set_title(f'{net.pipe_names[int(pipe)]} | source at {xi:.2f}L')
    x=np.arange(net.S)
    axes[1].plot(x,relative_times(clean)[0],'o-',color=TEAL,label='Graph first-arrival prior')
    axes[1].scatter(x,relative_times(noisy)[0],marker='x',color=PINK,s=65,
                    label=f'Timestamp error: σ = {sigma_ms:g} ms',zorder=3)
    axes[1].set_xticks(x,[_node_label(n) for n in net.sensors],rotation=45)
    axes[1].set(xlabel='Instrumented junction',ylabel='Arrival relative to earliest sensor (s)',
                title='A source location produces a timing pattern')
    axes[1].legend(frameon=False,fontsize=9)
    fig.text(.02,.015,f'Independent timing errors precede differencing; timestamps are rounded to '
             f'{1000*quantum:g} ms. The event onset cancels.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));return fig


def _feature_values(model,observations):
    import torch
    a=np.atleast_2d(np.asarray(observations,dtype=np.float32))
    device=next(model.parameters()).device
    with torch.inference_mode():
        tensor=torch.as_tensor(a,device=device)
        values=model.input_features(tensor)
        if isinstance(values,tuple): values=values[0]
    return values.detach().cpu().numpy()[0]


def feature_names(net):
    sensors=[_node_label(s) for s in net.sensors]
    return ([f'Landmark {s}' for s in sensors]+[f'Observed {s}' for s in sensors]+
            [f'Difference {s}' for s in sensors]+[f'Squared difference {s}' for s in sensors]+
            ['x coordinate','y coordinate','Degree','Boundary flag','Sensor flag','Local timing'])


def plot_features(net,model,observations,node_name=None):
    from model import predict
    name=node_name or net.sensors[0]; node=net.names.index(name)
    x=_feature_values(model,observations)[node]
    h=predict(model,np.atleast_2d(observations),return_embeddings=True)['embeddings'][0][0,node]
    colors=[TEAL]*net.S+[PINK]*net.S+[GOLD]*net.S+[GOLD]*net.S+[GREY]*5+[PINK]
    fig,axes=plt.subplots(2,1,figsize=(12,6.8),gridspec_kw={'height_ratios':[1.35,1]})
    axes[0].bar(np.arange(len(x)),x,color=colors,width=.85)
    centers=[(net.S-1)/2,(3*net.S-1)/2,(5*net.S-1)/2,(7*net.S-1)/2,4*net.S+2,4*net.S+5]
    axes[0].set_xticks(centers,['Graph\nlandmarks','Observed\ntimings','Difference','Squared\ndifference','Static\nproperties','Local\ntiming'],fontsize=9)
    for q in [net.S,2*net.S,3*net.S,4*net.S,4*net.S+5]:axes[0].axvline(q-.5,color='white',lw=2)
    axes[0].set(ylabel='Input value after scaling',title=f'{_node_label(name)}: {len(x)} supplied feature channels')
    axes[1].bar(np.arange(len(h)),h,color=NAVY)
    axes[1].set(xlabel='Learned embedding channel',ylabel='Embedding value',
                title=f'Encoder: the same input becomes a {len(h)}-dimensional learned vector')
    fig.text(.02,.015,'Input features have specified meanings. Learned channels are coordinates of a representation; '
             'they are not independently labelled hydraulic quantities.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));return fig


def plot_embeddings(net,model,observations,layer=2):
    from model import predict
    z=predict(model,np.atleast_2d(observations),return_embeddings=True)['embeddings'][int(layer)][0]
    z0=z-z.mean(axis=0)
    u,s,_=np.linalg.svd(z0,full_matrices=False);pc=u[:,:2]*s[:2]
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.3))
    im=axes[0].imshow(z[:,:12],aspect='auto',cmap='viridis')
    axes[0].set(xlabel='Learned channel (first 12)',ylabel='Original node index',
                title=f'Node embeddings | layer {int(layer)}')
    fig.colorbar(im,ax=axes[0],fraction=.04,pad=.02)
    axes[1].scatter(pc[:,0],pc[:,1],s=23,color=GREY,label='Other nodes')
    si=np.asarray(net.sensor_indices)
    axes[1].scatter(pc[si,0],pc[si,1],s=55,color=TEAL,label='Sensor nodes')
    for i in si:axes[1].annotate(_node_label(net.names[i]),pc[i],fontsize=7,xytext=(4,4),textcoords='offset points')
    axes[1].set(xlabel='Principal component 1',ylabel='Principal component 2',title='A 2D projection of the same vectors')
    axes[1].legend(frameon=False,fontsize=9)
    fig.text(.02,.015,'PCA is for display only. Its axes are not geographic coordinates; each layer has its own PCA projection.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));return fig


def plot_aggregation(net,model,observations,node_name=None,layer=0):
    from model import predict
    name=node_name or net.sensors[0];node=net.names.index(name)
    z=predict(model,np.atleast_2d(observations),return_embeddings=True)['embeddings'][int(layer)][0]
    row=model.normalized_adjacency.detach().cpu().numpy()[node]
    ids=np.flatnonzero(row);message=row@z
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),gridspec_kw={'width_ratios':[1,1.2,1.2]})
    colors=[GOLD if i==node else TEAL for i in ids]
    axes[0].bar(range(len(ids)),row[ids],color=colors)
    axes[0].set_xticks(range(len(ids)),[_node_label(net.names[i]) for i in ids],rotation=45)
    axes[0].set(ylabel='Normalised adjacency weight',title=f'Who contributes to {_node_label(name)}?')
    axes[1].imshow(z[ids,:8],aspect='auto',cmap='viridis')
    axes[1].set_yticks(range(len(ids)),[_node_label(net.names[i]) for i in ids])
    axes[1].set(xlabel='Embedding channel (first 8)',title='Vectors entering the aggregation')
    axes[2].plot(np.arange(8),z[node,:8],'o-',color=GOLD,label='Current node vector')
    axes[2].plot(np.arange(8),message[:8],'o-',color=TEAL,label='Weighted aggregate')
    axes[2].set(xlabel='Embedding channel',ylabel='Value',title='Receive and combine: ÂH')
    axes[2].legend(frameon=False,fontsize=8)
    fig.text(.02,.015,'Gold marks the self-connection. Symmetric degree normalisation is not a row-wise probability average. '
             'Learned weights and activation form the next update.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));return fig


def plot_training(history):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    epochs=history['epoch']
    axes[0].plot(epochs,history['train_loss'],color=TEAL,label='Training')
    axes[0].plot(epochs,history['val_loss'],color=PINK,label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Combined loss',title='Learn from training; select with validation')
    best=history.get('best_epoch')
    if best is not None:axes[0].axvline(best,color=GREY,ls='--',label=f'Selected epoch {best}')
    axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.15)
    if 'val_pipe_accuracy' in history:
        axes[1].plot(epochs,100*np.asarray(history['val_pipe_accuracy']),color=TEAL)
        axes[1].set(xlabel='Epoch',ylabel='Validation pipe accuracy (%)',ylim=(0,100),title='A diagnostic during training')
        axes[1].grid(alpha=.15)
    else:
        axes[1].axis('off')
    fig.text(.02,.015,'The checkpoint is selected by validation loss. Held-out source locations do not select the checkpoint.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));return fig


def plot_architecture(net,model):
    fig,ax=plt.subplots(figsize=(15,5.6));ax.set_xlim(0,15);ax.set_ylim(0,5.6);ax.axis('off')
    boxes=[(.1,2,2.1,1.5,f'Observed timings\n{net.S} sensors',PINK),
           (2.65,2,2.2,1.5,f'Node features\n{net.N} × {model.input_layer.in_features}',TEAL),
           (5.3,2,2.2,1.5,f'Encoder + {model.layers} updates\n{net.N} × {model.hidden}',NAVY),
           (7.95,2,2.15,1.5,'Endpoint vectors\n+ timing/pipe features',TEAL),
           (10.55,2,1.9,1.5,'Shared\npipe decoder',NAVY),
           (12.9,3.35,1.95,1.3,'Pipe scores\n→ probabilities',PINK),
           (12.9,.85,1.95,1.3,'Position fraction\n0 < ξ < 1',GOLD)]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.12',facecolor=color,edgecolor='none'))
        ax.text(x+w/2,y+h/2,label,color='white',ha='center',va='center',fontsize=10.5)
    arrows=[((2.25,2.75),(2.60,2.75)),((4.90,2.75),(5.25,2.75)),((7.55,2.75),(7.90,2.75)),
            ((10.15,2.75),(10.50,2.75)),((12.50,3),(12.85,3.85)),((12.50,2.50),(12.85,1.55))]
    for start,end in arrows:ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,lw=1.5,color=GREY))
    ax.text(.1,5.05,'From a timing pattern to a source inside a pipe',fontsize=18,fontweight='bold')
    ax.text(2.7,1.6,'Known graph landmarks\njoin the observed timings',fontsize=10,ha='left',va='top',color=TEAL)
    ax.text(5.35,1.6,'Shared node-update weights\ncombine connected nodes',fontsize=10,ha='left',va='top')
    ax.text(.1,.30,f'{net.P} original pipe columns; {net.source_pipe_count} eligible source pipes. Choose the highest-probability pipe, '
            'then use its fraction.\nThe unknown source position and event onset are not model inputs.',fontsize=10,va='center')
    fig.tight_layout();return fig


def metrics_table(report):
    import pandas as pd
    rows=[]
    for label,entry in report['results'].items():
        metrics=entry.get('GNN',entry)
        rows.append({'Evaluation':label,'Events':entry.get('events',np.nan),
                     'Localised events':entry.get('localised_events',entry.get('events',np.nan)),
                     'Pipe accuracy (%)':metrics['Pipe accuracy (%)'],
                     'Mean network distance error (m)':(
                         np.nan if metrics['Mean network distance error (m)'] is None
                         else metrics['Mean network distance error (m)'])})
    return pd.DataFrame(rows).set_index('Evaluation').round(2)


def plot_metrics(report):
    table=metrics_table(report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    short_labels={'Graph-prior held-out sources':'Graph prior\nheld-out sources', 'TSNet signals: clean':'TSNet\nclean signals', 'TSNet signals: noise and picking uncertainty':'TSNet\nnoisy signals + picks'}
    labels=[short_labels.get(str(x),str(x)) for x in table.index]
    for ax,key,color in zip(axes,['Pipe accuracy (%)','Mean network distance error (m)'],[TEAL,NAVY]):
        values=np.asarray(table[key],dtype=float)
        bars=ax.bar(range(len(table)),values,width=.55,color=color)
        for i,(bar,value) in enumerate(zip(bars,values)):
            if np.isfinite(value):
                ax.annotate(f'{value:.1f}',(bar.get_x()+bar.get_width()/2,value),
                            xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
            else:
                ax.text(i,.5,'N/A',transform=ax.get_xaxis_transform(),ha='center',color=PINK)
        ax.set_xticks(range(len(table)),labels,fontsize=9)
        ax.set_title(key);ax.margins(y=.20);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    axes[0].set_ylim(0,110)
    note='Distance error includes predictions on a different pipe. '
    if table.isna().any().any():
        note+='N/A: required arrivals are missing for some events; they count as incorrect in pipe accuracy.'
    else:
        note+='Pipe identity and physical distance answer different questions.'
    fig.text(.02,.015,note,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));return fig


def plot_event(net,model,dataset,index=0,show_raw=False):
    from signal_story import observe_event,event_record,localise_observation
    from graph_tools import network_error
    i=int(index);record=event_record(net,dataset,i)
    obs=observe_event(net,dataset,i)
    result=localise_observation(net,model,obs)
    p,xi=int(record['pipe']),float(record['xi'])
    predictions={}
    if result['available']:
        gp,gx=int(result['pipe']),float(result['xi'])
        error=float(np.asarray(network_error(net,p,xi,gp,gx)).reshape(-1)[0])
        predictions={'GNN estimate':(gp,gx,PINK,'o')}
    fig,axes=plt.subplots(2,2,figsize=(13,9),gridspec_kw={'height_ratios':[1.1,1]})
    draw_network(net,axes[0,0],source=(p,xi),predictions=predictions)
    axes[0,0].set_title(f'Event {i+1} | reveal the source',loc='left')
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper right')
    values=record['pressure_head'] if show_raw else record['signal']
    if show_raw:values=values-values[:1]
    for j,sensor in enumerate(net.sensors):
        axes[0,1].plot(record['time'],values[:,j],lw=1,label=_node_label(sensor))
    axes[0,1].set(xlabel='Time (s)',ylabel='Head change (m)',title='Raw pressure-head change' if show_raw else 'Burst response after background subtraction')
    axes[0,1].legend(ncol=5,fontsize=7,frameon=False)
    axes[1,0].axis('off')
    axes[1,0].set_title('Decode one physical location',loc='left')
    lines=[f'True pipe: {net.pipe_names[p]}',f'True position: {xi:.3f}L from its start node']
    if result['available']:
        lines += ['',f'GNN pipe: {net.pipe_names[gp]}',f'GNN position: {gx:.3f}L = {gx*net.lengths[gp]:.1f} m from its start',
                  f'Network distance error: {error:.1f} m']
        probs=np.asarray(result['probabilities']).reshape(-1)
        eligible=np.asarray(net.source_pipe_indices)
        top=eligible[np.argsort(probs[eligible])[-5:][::-1]]
        axes[1,1].barh(range(len(top)),probs[top],color=[GOLD if q==p else TEAL for q in top])
        axes[1,1].set_yticks(range(len(top)),[net.pipe_names[q] for q in top]);axes[1,1].invert_yaxis()
        axes[1,1].set(xlabel='Model probability',title='Which pipes receive the largest scores?')
        axes[1,1].set_xlim(0,max(.35,probs[top].max()*1.15))
    else:
        lines += ['',result.get('message','An arrival was not detected; no prediction is made.')]
        axes[1,1].axis('off')
    axes[1,0].text(0,.93,'\n'.join(lines),transform=axes[1,0].transAxes,fontsize=11,va='top',linespacing=1.65)
    fig.text(.02,.015,'The GNN uses independently picked relative arrival times from 100 Hz signals. '
             'The true position is used only to display and score the result. Probabilities are not calibrated uncertainty.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));return fig


def _show(fig):
    from IPython.display import display
    display(fig);plt.close(fig)


def panel(widget,title='Explore interactively'):
    import ipywidgets as w
    box=w.Accordion(children=[widget],selected_index=None)
    box.set_title(0,title);return box


def _live_tabs(controls,renderers,description=''):
    """Lazy rendering keeps hidden tabs light and gives all controls one state."""
    import ipywidgets as w
    outputs=[w.Output() for _ in renderers]
    tab=w.Tab(children=outputs)
    for i,name in enumerate(renderers):tab.set_title(i,name)
    row=w.HBox(list(controls.values()),layout=w.Layout(flex_flow='row wrap',gap='12px'))
    box=w.VBox(([w.HTML(description)] if description else [])+[row,tab])
    def update(change=None):
        i=tab.selected_index
        if i is None:return
        with outputs[i]:
            outputs[i].clear_output(wait=True)
            renderers[list(renderers)[i]]()
    for item in controls.values():
        if 'value' in item.traits():item.observe(update,names='value')
    tab.observe(update,names='selected_index')
    # Named access is used by the notebook's callback smoke check.
    box._tutorial_controls=controls;box._tutorial_tabs=tab;box._tutorial_render=update
    update();return box


def network_explorer(net):
    import ipywidgets as w
    from IPython.display import display
    nodes=w.Checkbox(value=False,description='Node IDs')
    pipes=w.Checkbox(value=False,description='Pipe IDs')
    node=w.Dropdown(options=net.names,value=net.sensors[0],description='Focus:')
    hops=w.IntSlider(value=1,min=1,max=3,description='Hops:',continuous_update=False)
    return _live_tabs({'nodes':nodes,'pipes':pipes,'node':node,'hops':hops},
        {'Full network':lambda:_show(plot_network(net,nodes.value,pipes.value)),
         'One neighbourhood':lambda:_show(plot_adjacency(net,node.value,hops.value)),
         'Pipe properties':lambda:display(link_table(net,node.value,hops.value))},
        '<b>Read the same network at two scales.</b> Labels preserve the original INP IDs. '
        'The pipe table uses physical lengths and propagation speeds, not distances on the drawing.')


def adjacency_explorer(net):
    import ipywidgets as w
    from IPython.display import display
    node=w.Dropdown(options=net.names,value='J84' if 'J84' in net.names else net.names[0],description='Node:')
    hops=w.IntSlider(value=1,min=1,max=3,description='Hops:',continuous_update=False)
    return _live_tabs({'node':node,'hops':hops},
        {'Connectivity':lambda:_show(plot_adjacency(net,node.value,hops.value,'binary')),
         'Add self-links':lambda:_show(plot_adjacency(net,node.value,hops.value,'self')),
         'Normalise':lambda:_show(plot_adjacency(net,node.value,hops.value,'normalised')),
         'Travel costs':lambda:display(link_table(net,node.value,hops.value))},
        '<b>Two different uses of a graph.</b> Adjacency tells the GNN which nodes exchange messages. '
        'The travel-cost table assigns L/a seconds to physical pipes. A message-passing update does not simulate water hammer.')


def source_explorer(net,model=None):
    import ipywidgets as w
    from IPython.display import display
    import pandas as pd
    from graph_tools import travel_times,relative_times,add_timing_noise,network_error
    eligible=[int(i) for i in net.source_pipe_indices]
    preferred=eligible[0]
    pipe=w.Dropdown(options=[(net.pipe_names[i],i) for i in eligible],value=preferred if preferred in eligible else eligible[0],description='Pipe:')
    position=w.FloatSlider(value=.4,min=.02,max=.98,step=.01,description='Fraction ξ:',continuous_update=False)
    noise=w.FloatSlider(value=10,min=0,max=100,step=1,description='Jitter (ms):',continuous_update=False)
    def observation():
        clean=travel_times(net,[pipe.value],[position.value])
        return clean,relative_times(add_timing_noise(clean,noise.value/1000,.010,np.random.default_rng(10)))
    def table():
        clean,y=observation()
        display(pd.DataFrame({'Sensor':net.sensors,'Travel time (s)':clean[0],
                'Clean relative (s)':relative_times(clean)[0],'Observed relative (s)':y[0]}).round(4))
    def prediction():
        from model import predict
        _,y=observation();g=predict(model,y)
        p,x=int(g['pipe'][0]),float(g['xi'][0])
        error=float(np.asarray(network_error(net,pipe.value,position.value,p,x)).reshape(-1)[0])
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        draw_network(net,axes[0],source=(pipe.value,position.value),predictions={'GNN estimate':(p,x,PINK,'o')})
        axes[0].legend(frameon=False,fontsize=8);axes[0].set_title('True source and GNN location')
        probs=g['probabilities'][0];top=np.asarray(eligible)[np.argsort(probs[eligible])[-5:][::-1]]
        axes[1].barh(range(len(top)),probs[top],color=[GOLD if q==pipe.value else TEAL for q in top])
        axes[1].set_yticks(range(len(top)),[net.pipe_names[q] for q in top]);axes[1].invert_yaxis()
        axes[1].set(xlabel='Model probability',title='Five largest pipe probabilities')
        fig.tight_layout();_show(fig)
        print(f'GNN: {net.pipe_names[p]}, fraction {x:.3f}L; network distance error {error:.1f} m.')
        print('These probabilities are not calibrated confidence intervals.')
    tabs={'Travel-time pattern':lambda:_show(plot_travel(net,pipe.value,position.value,noise.value)),
          'Sensor values':table}
    if model is not None:tabs['GNN location']=prediction
    return _live_tabs({'pipe':pipe,'position':position,'noise':noise},tabs,
        'Move a source continuously within an eligible pipe. The random seed stays fixed while you change the timing-error scale; '
        'timestamps use the 100 Hz grid.')


def feature_explorer(net,model,observations):
    import ipywidgets as w
    import pandas as pd
    from IPython.display import display
    node=w.Dropdown(options=net.names,value=net.sensors[0],description='Node:')
    def table():
        x=_feature_values(model,observations)[net.names.index(node.value)]
        display(pd.DataFrame({'Feature':feature_names(net),'Value':x}).round(5))
    return _live_tabs({'node':node},
        {'Features → embedding':lambda:_show(plot_features(net,model,observations,node.value)),
         'Inspect every input':table},
        'Known travel-time landmarks and observed timing values are available at inference. '
        'The source label is used in the training loss, not in these input features.')


def embedding_explorer(net,model,observations):
    import ipywidgets as w
    layer=w.IntSlider(min=0,max=model.layers,value=model.layers,description='Layer:',continuous_update=False)
    node=w.Dropdown(options=net.names,value=net.sensors[0],description='Inspect node:')
    return _live_tabs({'layer':layer,'node':node},
        {'Embeddings and PCA':lambda:_show(plot_embeddings(net,model,observations,layer.value)),
         'Receive and combine':lambda:_show(plot_aggregation(net,model,observations,node.value,layer.value))},
        'Layer 0 is the encoder output. Each graph update combines neighbouring representations. '
        'The aggregation tab shows what Â would collect from the selected layer; at the final layer it is an inspection, not an extra model layer.')


def sensor_explorer(net):
    import ipywidgets as w
    from IPython.display import display
    import pandas as pd
    from graph_tools import load_network,ambiguity_report
    junctions=[n for i,n in enumerate(net.names) if i not in net.boundary_indices]
    select=w.SelectMultiple(options=junctions,value=tuple(net.sensors),rows=9,description='Sensors:',layout=w.Layout(width='340px'))
    button=w.Button(description='Inspect selected layout',button_style='info')
    out=w.Output();state={'net':net}
    def show_map():_show(plot_network(state['net'],False,False))
    def show_ambiguity():
        report=ambiguity_report(state['net'])
        constant=report['constant_signature_pipes']
        print(f'{len(constant)} eligible source pipes have a constant relative-arrival signature.')
        display(pd.DataFrame({'Pipe with unidentifiable position':constant}))
        print('Moving within such a pipe adds the same delay at every sensor; relative timing removes that delay.')
    tabs=_live_tabs({}, {'Sensor map':show_map,'Observability':show_ambiguity})
    def clicked(_):
        button.disabled=True
        try:
            with out:
                out.clear_output(wait=True)
                try:
                    state['net']=load_network(sensors=select.value,inp_path=net.inp_path,wave_speed=net.wave_speed,
                        prior_dt=net.metadata.get('prior_dt_requested_seconds',.00104045),
                        calibrated=net.metadata.get('prior_calibrated',True),min_source_pipe_length=net.min_source_pipe_length)
                    print('Layout inspected. The notebook’s active network and trained GNN are unchanged.')
                    tabs._tutorial_render()
                except ValueError as exc:print(str(exc))
        finally:button.disabled=False
    button.on_click(clicked)
    box=w.VBox([w.HTML('Hold Ctrl (Windows) or Command (Mac) to select multiple junctions. '
        'Changing the model’s sensor layout requires new training.'),w.HBox([select,button]),out,tabs])
    box._tutorial_controls={'sensors':select,'inspect':button};box._tutorial_tabs=tabs._tutorial_tabs
    box._tutorial_render=tabs._tutorial_render;return box


def event_explorer(net,model,dataset):
    import ipywidgets as w
    event=w.Dropdown(options=[(f'Event {i+1}',i) for i in range(len(dataset['pipe']))],description='Event:')
    return _live_tabs({'event':event},
        {'Source and model outputs':lambda:_show(plot_event(net,model,dataset,event.value)),
         'Raw pressure context':lambda:_show(plot_event(net,model,dataset,event.value,True))},
        'Each event is a held-out TSNet burst. First arrivals are repicked from the stored 100 Hz signals before inference.')


def burst_lab(net,model):
    """A fresh hydraulic simulation runs only after an explicit button click."""
    import ipywidgets as w
    eligible=[net.pipe_names[int(i)] for i in net.source_pipe_indices]
    pipe=w.Dropdown(options=eligible,value=eligible[0],description='Pipe:')
    xi=w.FloatSlider(value=.4,min=.05,max=.95,step=.01,description='Fraction ξ:',continuous_update=False)
    rise=w.FloatSlider(value=.02,min=.01,max=.25,step=.01,description='Opening (s):',continuous_update=False)
    size=w.FloatSlider(value=.001/np.sqrt(40.),min=.00002,max=.0005,step=.00001,readout_format='.5f',description='Coefficient:',continuous_update=False)
    button=w.Button(description='Run a new TSNet burst',button_style='warning',icon='play');out=w.Output()
    def clicked(_):
        button.disabled=True
        try:
            with out:
                out.clear_output(wait=True)
                print('Running burst and no-burst reference simulations. This may take several minutes.',flush=True)
                try:
                    from tsnet_tools import simulate_burst
                    r=simulate_burst(pipe.value,xi.value,sensors=net.sensors,inp_path=net.inp_path,
                        rise_time=rise.value,coefficient=size.value,wave_speed=net.wave_speed,
                        duration=3.,dt=.00104045)
                    p=net.pipe_names.index(pipe.value)
                    # Put new traces on the same 100 Hz observation grid as
                    # the prepared teaching examples before arrival picking.
                    source_time=np.asarray(r['time'])
                    time=np.arange(np.ceil(source_time[0]/.010),
                                   np.floor(source_time[-1]/.010)+1)*.010
                    ds={key:np.column_stack([np.interp(time,source_time,column)
                            for column in np.asarray(r[key]).T])[None]
                        for key in ['heads','signal']}
                    ds.update(time=time,node_names=r['node_names'],pipe=np.array([p]),
                        xi=np.array([xi.value]),pipe_names=np.array([pipe.value]),elevations=r['elevations'],
                        metadata_json=json.dumps([r['metadata']]))
                    _show(plot_event(net,model,ds))
                    print(f"Completed in {r['metadata']['runtime_s']:.1f} s; solver step {1000*r['actual_dt']:.2f} ms.")
                    print('The GNN uses picked relative arrival times. Coefficient units: m³/s/√m.')
                    if r['metadata']['minimum_pressure_head_m']<0:
                        print('Negative pressure occurred. Reduce the burst size; this solver does not represent cavitation.')
                except Exception as exc:
                    print(f'Simulation needs attention: {type(exc).__name__}: {exc}')
        finally:button.disabled=False
    button.on_click(clicked)
    box=w.VBox([w.HTML('Optional exploration after the lecture. No solver is started when this cell runs.'),
                w.HBox([pipe,xi]),w.HBox([rise,size]),button,out])
    box._tutorial_controls={'pipe':pipe,'position':xi,'rise':rise,'coefficient':size,'run':button}
    return box
