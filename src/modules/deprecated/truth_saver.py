import sqlite3
from pathlib import Path
import pandas as pd
import pickle


class TruthSaver:
    def __init__(self, config, files_and_dirs, events):
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.events = events
        self.targets = ['event_no'] + config['targets']

        self.test_set_db_path = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_set.db')
        print(self.test_set_db_path)
        self.predictions_db_path = files_and_dirs['dbs'].joinpath('predictions.db')

    def save_truths_to_db(self):
        query = 'SELECT {targets} FROM scalar'.format(
            targets=', '.join(self.targets)
        )
        with sqlite3.connect(self.test_set_db_path) as con:
            truths = pd.read_sql_query(query, con)
        truths.set_index('event_no', inplace=True)
        with sqlite3.connect(self.predictions_db_path) as con:
            truths.to_sql('truth', con=con, if_exists='replace')
