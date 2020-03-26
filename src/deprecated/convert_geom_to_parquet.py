import pandas as pd
from pathlib import Path

geom_in_file = Path().home().joinpath('CubeFlowData').joinpath('geom').joinpath('GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.hdf5')
geom_out_file = Path().home().joinpath('CubeFlowData').joinpath('geom').joinpath('GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.gzip')
geom = pd.read_hdf(geom_in_file, key='pmt_geom')
geom.to_parquet(geom_out_file, compression='gzip', engine='fastparquet')