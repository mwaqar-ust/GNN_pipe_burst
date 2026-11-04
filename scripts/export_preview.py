"""Export the executed notebook as a static, self-contained-figure HTML preview."""
from pathlib import Path
import copy
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]

def main():
    nb = nbformat.read(ROOT/'IAHR_GNN_HKUST_Tutorial.ipynb',as_version=4)
    if any(c.cell_type=='code' and c.execution_count is None for c in nb.cells):
        raise RuntimeError('Execute the notebook before exporting its preview.')
    preview = copy.deepcopy(nb)
    preview.metadata.pop('widgets',None)
    for cell in preview.cells:
        if cell.cell_type != 'code': continue
        cell.outputs = [o for o in cell.outputs if 'application/vnd.jupyter.widget-view+json' not in o.get('data',{})]
    preview.cells.insert(0,nbformat.v4.new_markdown_cell(
        '> **Static reading preview.** Figures and explanations are saved here. '
        'Open the notebook from the complete project folder in Jupyter to use the interactive controls. '
        'Equation rendering may require an internet connection for MathJax.'))
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body,_ = exporter.from_notebook_node(preview)
    target = ROOT/'IAHR_GNN_HKUST_Tutorial_Preview.html'
    target.write_text(body,encoding='utf-8')
    print(f'Saved {target.name} ({target.stat().st_size:,} bytes).')

if __name__=='__main__':main()
