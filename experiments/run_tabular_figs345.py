"""Paper Figs 3A/3B, 4, 5 and Table 1 on tabular data.

Usage: python experiments/run_tabular_figs345.py {inventory,screen,figures,fig4,all} [--cases ...] [--device cpu]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')

from wvalue import tabular as tb

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['inventory', 'screen', 'figures', 'fig4', 'all'])
    p.add_argument('--uci-dir', default=os.path.join(REPO_ROOT, 'UCI_data'),
                   help='folder holding <name>/<name>_R.dat (default: <repo>/UCI_data, git-ignored)')
    p.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'results_tabular'),
                   help='parent folder for runs (default: <repo>/results_tabular, git-ignored)')
    p.add_argument('--run-name', default='tabular_figs345_v1')
    p.add_argument('--no-resume', action='store_true', help='recompute instead of resuming')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--cases', nargs='+', default=None,
                   help='override automatic selection, e.g. uci:spambase openml:pol')
    p.add_argument('--seeds', nargs='+', type=int, default=None, help='screen seeds (default 0-4)')
    p.add_argument('--full-seeds', nargs='+', type=int, default=None,
                   help='seeds for figures/fig4 (mean/std bands if more than one)')
    p.add_argument('--regimes', nargs='+', default=None, choices=['fixed', 'tuned'])
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--finetune-epochs', type=int, default=None)
    p.add_argument('--max-ntrain', type=int, default=None, help='skip datasets with larger training sets')
    p.add_argument('--no-openml', action='store_true', help='skip OpenML regression datasets (no network)')
    p.add_argument('--no-uci', action='store_true')
    return p.parse_args(argv)

def build_config(a):
    cfg = tb.TabularConfig()
    if a.seeds is not None:
        cfg.SCREEN_SEEDS = a.seeds
    if a.full_seeds is not None:
        cfg.FULL_SEEDS = a.full_seeds
    if a.regimes is not None:
        cfg.REGIMES = a.regimes
    if a.epochs is not None:
        cfg.EPOCHS = a.epochs
    if a.finetune_epochs is not None:
        cfg.FINETUNE_EPOCHS = a.finetune_epochs
    if a.max_ntrain is not None:
        cfg.MAX_NTRAIN = a.max_ntrain
    if a.cases:
        cfg.MANUAL_CASES = a.cases
    if a.no_openml:
        cfg.INCLUDE_OPENML_REGRESSION = False
    if a.no_uci:
        cfg.INCLUDE_UCI = False
    return cfg

def main(argv=None):
    a = parse_args(argv)
    resume = not a.no_resume
    cfg = build_config(a)
    tb.configure(cfg, device=a.device, uci_dir=a.uci_dir)
    print('device:', tb.DEVICE)
    paths = tb.RunPaths(os.path.join(a.out_dir, a.run_name))
    paths.check_config(resume=resume)
    print('output root:', paths.root)

    cmd = a.command
    if cmd in ('inventory', 'screen', 'all'):
        _, eligible = tb.run_inventory(paths)
    if cmd in ('screen', 'all'):
        rows = tb.run_screen(paths, eligible, resume=resume)
        _, _, selected = tb.summarize_screen(paths, rows)
    if cmd in ('figures', 'fig4', 'all'):
        selected = tb.selected_cases(paths)
        print('Selected datasets:', selected)
    if cmd in ('figures', 'all'):
        full = tb.run_full_figures(paths, selected, resume=resume)
        tb.make_table1(paths, full)
    if cmd in ('fig4', 'all'):
        tb.run_fig4(paths, selected, resume=resume)
    print('Done. Outputs in', paths.root)

if __name__ == '__main__':
    main()
