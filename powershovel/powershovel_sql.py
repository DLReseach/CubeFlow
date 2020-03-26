import sqlite3
import pandas as pd
from pathlib import Path


class DbHelper:
    def __init__(self, prediction_file, test_set_file):
        super(DbHelper).__init__()
        self.con_pred = sqlite3.connect(prediction_file)
        self.con_sequential = sqlite3.connect(test_set_file)

    def get_predictions(self, run_name, comparison_name, event_no):
        query = 'SELECT * FROM \'{}\' WHERE event_no = {}'.format(run_name, event_no)
        event_predictions = pd.read_sql_query(query, self.con_pred)
        query = 'SELECT * FROM truth WHERE event_no = {}'.format(event_no)
        event_truths = pd.read_sql_query(query, self.con_pred)
        query = 'SELECT * FROM sequential WHERE event_no = {}'.format(event_no)
        event_sequential = pd.read_sql_query(query, self.con_sequential)
        if comparison_name == 'retro_crs_prefit':
            query = 'SELECT * FROM retro_crs_prefit WHERE event_no = {}'.format(event_no)
        else:
            query = 'SELECT * FROM \'{}\' WHERE event_no = {}'.format(comparison_name, event_no)
        event_comparison = pd.read_sql_query(query, self.con_pred)
        query = 'SELECT * FROM meta WHERE event_no = {}'.format(event_no)
        event_meta = pd.read_sql_query(query, self.con_sequential)
        return event_predictions, event_truths, event_sequential, event_comparison, event_meta
