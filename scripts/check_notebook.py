"""Execute the complete prepared tutorial; optional training stays disabled.

Default uses a fresh Jupyter kernel. --in-process supports restricted hosts
without kernel sockets/ipykernel. --write saves verified outputs in the main
notebook; otherwise the executed copy stays in outputs/.
"""
from pathlib import Path
import argparse
import ast
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def inspect_source(nb):
    retrain = []
    for cell in nb.cells:
        if cell.cell_type != 'code': continue
        tree = ast.parse(cell.source)
        for item in tree.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == 'RETRAIN':
                        retrain.append(isinstance(item.value, ast.Constant) and item.value.value is False)
    if retrain != [True]:
        raise RuntimeError('Use exactly one RETRAIN=False setup assignment for the prepared notebook check.')

def execute_inprocess(nb):
    import nbformat
    import matplotlib
    matplotlib.use('Agg')
    import ipywidgets as widgets
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.pylabtools import select_figure_formats
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    select_figure_formats(shell, formats=['png'])
    sys.path.insert(0, str(ROOT))
    for count, cell in enumerate((c for c in nb.cells if c.cell_type=='code'), 1):
        outputs = []
        seen = set()
        # Without a kernel, Output widgets cannot isolate clear_output messages.
        # Capture each top-level statement so an explorer cannot erase the static
        # figure displayed earlier in the cell. All statements share shell state.
        statements = ast.parse(cell.source).body
        for index, statement in enumerate(statements):
            source = ast.get_source_segment(cell.source, statement)
            previous_interactivity = shell.ast_node_interactivity
            try:
                if index < len(statements) - 1:
                    shell.ast_node_interactivity = 'none'
                with capture_output(stdout=True, stderr=True, display=True) as captured:
                    result = shell.run_cell(source, store_history=False)
            finally:
                shell.ast_node_interactivity = previous_interactivity
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f'Code cell {count} failed:\n{captured.stdout}\n{captured.stderr}')
            for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                value = value.replace('\x1b[2K','').replace('\r','')
                if value.strip(): outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
            widget_outputs = [rich for rich in captured.outputs
                              if 'application/vnd.jupyter.widget-view+json' in rich.data]
            if widget_outputs:
                # Nested explorer plots belong inside their widget, not the
                # notebook's top-level static reading sequence.
                rich_outputs = widget_outputs
            else:
                rich_outputs = captured.outputs
            for rich in rich_outputs:
                png = rich.data.get('image/png')
                if png and png in seen: continue
                if png: seen.add(png)
                outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = count
        print(f'Executed code cell {count}', flush=True)
    nb.metadata['widgets'] = {'application/vnd.jupyter.widget-state+json': widgets.Widget.get_manager_state()}
    return nb

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-process', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--kernel', default='iahr-gnn-hkust-tutorial')
    args = parser.parse_args()
    import nbformat
    source = ROOT/'IAHR_GNN_HKUST_Tutorial.ipynb'
    nb = nbformat.read(source, as_version=4)
    inspect_source(nb)
    started = time.perf_counter()
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        if args.in_process:
            nb = execute_inprocess(nb)
        else:
            from nbclient import NotebookClient
            nb = NotebookClient(nb,timeout=240,kernel_name=args.kernel,
                resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        os.chdir(previous)
    nbformat.validate(nb)
    errors = [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
    if errors: raise RuntimeError(f'Notebook contains {len(errors)} error outputs.')
    output = ROOT/'outputs'
    output.mkdir(exist_ok=True)
    target = source if args.write else output/'notebook_check.ipynb'
    nbformat.write(nb, target)
    import numpy, torch
    report = {'code_cells':sum(c.cell_type=='code' for c in nb.cells),
        'error_outputs':0,'seconds':round(time.perf_counter()-started,2),
        'execution':'IPython in-process' if args.in_process else 'fresh Jupyter kernel',
        'optional_training':False,'optional_simulation':False,
        'platform':sys.platform,'python':sys.version.split()[0],
        'numpy':numpy.__version__,'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
    (output/'notebook_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Saved',target.name)

if __name__=='__main__': main()
