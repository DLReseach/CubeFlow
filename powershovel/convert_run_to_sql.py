import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

SQL_FILE = Path('/mnt/c/Users/MadsEhrhorn/Downloads/test_set_backup.db')
# SQL_FILE = Path().home().joinpath('sqlite/test_set.db')
RUN_PATH = Path().home().joinpath('runs')
engine = create_engine('sqlite:///' + str(SQL_FILE), echo=False)
connection = engine.connect()
RUN = 'ingenious-poodle'

PREDICTIONS_DF = pd.read_parquet(str(RUN_PATH.joinpath(RUN).joinpath('run_dataframe.gzip')))
PREDICTIONS_DF.set_index('event', inplace=True)

PREDICTIONS_DF.to_sql(RUN, con=connection, if_exists='replace', index=True)
