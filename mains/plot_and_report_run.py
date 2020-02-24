from pathlib import Path

from src.modules.utils import get_project_root
from src.modules.resolution_comparison import ResolutionComparison

files_and_dirs = {}
files_and_dirs['run_root'] = Path().home().joinpath('cnn1d_3_lin_3_energy_0_to_4_muon_2020-02-24.spicy-agouti')

comparer_config = {
    'dom_plots': True,
    'use_train_dists': False,
    'only_use_metrics': None,
    'legends': True,
    'reso_hists': True,
    'use_bootstrapped': True,
    'wandb': False
}
comparison_metrics = [
    'azimuth',
    'zenith',
    'energy',
    'time'
]

comparer = ResolutionComparison(comparison_metrics, files_and_dirs, comparer_config, reporter=None)
comparer.testing_ended()
