import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

SQL_FILE = Path('/mnc/c/Users/MadsEhrhorn/Downloads').joinpath('test_set_backup.db')
RUN_PATH = Path().home().joinpath('runs')
# engine = create_engine('sqlite:///' + str(SQL_FILE), echo=False)
# connection = engine.connect()
RUN = 'vehement-emu'

PREDICTIONS_DF = pd.read_parquet(str(RUN_PATH.joinpath(RUN).joinpath('prediction_dataframe_parquet_for_db.gzip')))
OWN_ERROR_DF = pd.read_parquet(str(RUN_PATH.joinpath(RUN).joinpath('own_error_dataframe_parquet.gzip')))
OPP_ERROR_DF = pd.read_parquet(str(RUN_PATH.joinpath(RUN).joinpath('opponent_error_dataframe_parquet.gzip')))

print(PREDICTIONS_DF.head())
print(OWN_ERROR_DF.columns)
print(OPP_ERROR_DF.columns)
