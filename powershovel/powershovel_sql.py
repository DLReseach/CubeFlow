import sqlite3
import pandas as pd
from pathlib import Path


class DbHelper:
    def __init__(self, sql_file):
        super(DbHelper).__init__()
        self.con = sqlite3.connect(sql_file)
        self.cursor = self.con.cursor()

    def get_predictions(self, run_name, comparison_name, event_no):
        query = 'SELECT * FROM \'{}\' WHERE event = {}'.format(run_name, event_no)
        event_predictions = pd.read_sql_query(query, self.con)
        query = 'SELECT * FROM scalar WHERE event_no = {}'.format(event_no)
        event_scalars = pd.read_sql_query(query, self.con)
        query = 'SELECT * FROM sequential WHERE event_no = {}'.format(event_no)
        event_sequential = pd.read_sql_query(query, self.con)
        if comparison_name == 'retro_crs_prefit':
            query = 'SELECT * FROM \'{}\' WHERE event = {}'.format(run_name, event_no)
        else:
            query = 'SELECT * FROM \'{}\' WHERE event = {}'.format(comparison_name, event_no)
        event_comparison = pd.read_sql_query(query, self.con)
        query = 'SELECT * FROM meta WHERE event_no = {}'.format(event_no)
        event_meta = pd.read_sql_query(query, self.con)
        return event_predictions, event_scalars, event_sequential, event_comparison, event_meta
