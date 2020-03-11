import sqlite3
import pandas as pd
from pathlib import Path


class DbHelper:
    def __init__(self, sql_file):
        super(DbHelper).__init__()
        self.con = sqlite3.connect(sql_file)
        self.cursor = self.con.cursor()

    def get_predictions(self, run_name, event_no):
        query = 'SELECT * FROM \'{}\' WHERE event = {}'.format(run_name, event_no)
        event_predictions = pd.read_sql_query(query, self.con)
        return event_predictions
