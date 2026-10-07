"""Download the 121-dataset UCI collection into UCI_data/ (git-ignored)."""
import os
import sys
import shutil
import tarfile
import zipfile
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

GDRIVE_ID = '1Z3tEzCmR-yTvn1ZlAXaeAuVB5a9oCAkk'

def download(archive):
    if os.path.exists(archive) and os.path.getsize(archive) > 50e6:
        return
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
    gdown.download(id=GDRIVE_ID, output=archive, quiet=False)
    if not os.path.exists(archive):
        raise RuntimeError('Download failed. Fetch the archive manually (see README) and unpack it '
                           'so that UCI_data/<name>/<name>_R.dat exists.')

def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, 'r:*') as tf:
            tf.extractall(dest, filter='data') if sys.version_info >= (3, 12) else tf.extractall(dest)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with open(archive, 'rb') as fh:
            head = fh.read(300)
        raise RuntimeError(f'Unrecognised archive format. First bytes: {head!r}')

def find_dataset_dirs(root):
    found = {}
    for r, _dirs, files in os.walk(root):
        name = os.path.basename(r)
        if f'{name}_R.dat' in files or f'{name}_train_R.dat' in files:
            found[name] = r
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uci-dir', default=os.path.join(REPO_ROOT, 'UCI_data'))
    ap.add_argument('--tmp-dir', default=os.path.join(REPO_ROOT, 'UCI_data_tmp'))
    ap.add_argument('--keep-tmp', action='store_true')
    a = ap.parse_args()

    from wvalue.datasets import list_uci_datasets, load_uci_dataset
    if os.path.isdir(a.uci_dir) and len(list_uci_datasets(a.uci_dir)) > 100:
        print(f'{a.uci_dir} already holds {len(list_uci_datasets(a.uci_dir))} datasets; nothing to do.')
        return

    os.makedirs(a.tmp_dir, exist_ok=True)
    os.makedirs(a.uci_dir, exist_ok=True)
    archive = os.path.join(a.tmp_dir, 'data.tar.gz')
    extract_dir = os.path.join(a.tmp_dir, 'extracted')
    download(archive)
    print(f'Archive: {os.path.getsize(archive) / 1e6:.1f} MB')
    if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
        extract(archive, extract_dir)

    dirs = find_dataset_dirs(extract_dir)
    print(f'Found {len(dirs)} datasets')
    if not dirs:
        raise RuntimeError('No <name>/<name>_R.dat folders found; archive layout differs from expected.')
    copied = skipped = 0
    for name, src in sorted(dirs.items()):
        dst = os.path.join(a.uci_dir, name)
        if os.path.isdir(dst) and any(f.endswith('_R.dat') for f in os.listdir(dst)):
            skipped += 1
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied += 1
    print(f'Copied {copied}, already present {skipped} -> {a.uci_dir}')

    names = list_uci_datasets(a.uci_dir)
    print(f'list_uci_datasets sees {len(names)} datasets')
    for t in ['spambase', 'iris']:
        if t in names:
            _tl, _vl, cfg = load_uci_dataset(t, uci_dir=a.uci_dir)
            print(f"  {t}: d={cfg['input_size']}, classes={cfg['num_classes']}, "
                  f"n_train={cfg['n_train']}, n_test={cfg['n_test']}")
    if not a.keep_tmp:
        shutil.rmtree(a.tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
