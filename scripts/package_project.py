"""Build the complete download ZIP, excluding local environments and run output."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'outputs','__pycache__','.ipynb_checkpoints','.git','.venv','venv','env'}

def files_for_release():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):continue
        if path.name=='MANIFEST.sha256':continue
        if path.suffix in {'.pyc','.log','.tmp','.pt','.rpt','.bin','.out'}:continue
        if '.redaction.' in path.name or '.tmp.' in path.name:continue
        if path.stat().st_size>=90_000_000:
            raise RuntimeError(f'File is too large for this teaching repository: {relative}')
        yield path

def main():
    required=['IAHR_GNN_HKUST_Tutorial.ipynb','IAHR_GNN_HKUST_Tutorial_Preview.html','README.md',
              'models/gnn_demo.npz','data/tsnet_test.npz','data/project_verification.json',
              'environment.yml','requirements.txt','setup_windows.bat','setup_unix.sh']
    for name in required:
        if not (ROOT/name).is_file():raise FileNotFoundError(name)
    nb=json.loads((ROOT/'IAHR_GNN_HKUST_Tutorial.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            if cell['execution_count'] is None or any(o['output_type']=='error' for o in cell['outputs']):
                raise RuntimeError('The delivered notebook must have a successful complete execution.')
    bat=ROOT/'setup_windows.bat'
    bat.write_bytes(bat.read_text(encoding='utf-8').replace('\n','\r\n').encode('utf-8'))
    files=list(files_for_release())
    manifest=ROOT/'MANIFEST.sha256'
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf-8')
    target=ROOT.parent/'IAHR_GNN_HKUST_Tutorial.zip'
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in [*files,manifest]:bundle.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:raise RuntimeError('Archive CRC check failed.')
    temporary.replace(target)
    print(f'Created {target.name}: {len(files)+1} files, {target.stat().st_size:,} bytes.')

if __name__=='__main__':main()
