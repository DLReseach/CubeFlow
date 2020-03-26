from pathlib import Path
import pickle
from src.modules.histogram_calculator import HistogramCalculator

masks_name = 'dom_interval_SplitInIcePulses_min0_max200-muon_neutrino'
run_name = 'pompous-puma'
dirs = {}
dirs['dbs'] = Path().home().joinpath('CubeFlowData').joinpath('dbs')

HistogramCalculator(masks_name, run_name, dirs)

with open(dirs['dbs'].joinpath(masks_name).joinpath('histograms.pkl'), 'rb') as f:
    test = pickle.load(f)

print(len(test['meta']['events'][0]))
