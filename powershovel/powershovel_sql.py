import sqlite3
import pandas as pd
from pathlib import Path


class DbHelper:
    def __init__(self, sql_file):
        super(DbHelper).__init__()
        self.con = sqlite3.connect(sql_file)
        self.cursor = self.con.cursor()
        self.get_tables()

    def get_tables(self):
        self.runs = []
        self.cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\";')
        tables = self.cursor.fetchall()
        for table in tables:
            if table[0] not in ['sequential', 'scalar', 'meta']:
                self.runs.append(table[0])

    def get_run_metadata(self, run_name):
        run_metadata = {}
        self.cursor.execute('SELECT DISTINCT(true_primary_energy_bin) from \'{}\' ORDER BY true_primary_energy_bin'.format(run_name))
        run_metadata['true_primary_energy_bin'] = list(map(lambda x: x[0], self.cursor.fetchall()))

        self.cursor.execute('SELECT DISTINCT(event_length_bin) from \'{}\' ORDER BY event_length_bin'.format(run_name))
        run_metadata['event_length_bin'] = list(map(lambda x: x[0], self.cursor.fetchall()))        

        self.cursor.execute('PRAGMA table_info(\'{}\')'.format(run_name))
        metric_names = list(map(lambda x: x[1], self.cursor.fetchall()))
        metric_names = [name for name in metric_names if name.startswith('opponent')]
        run_metadata['metric_names'] = [name.replace('opponent_', '').replace('_error', '') for name in metric_names]
        return run_metadata

    def get_metric_errors(self, metric, selected_run, selected_comparison, bin_in, bin_name):
        own_metric = 'predicted_' + metric + '_error'
        query = 'SELECT {}, {}, {}, {}, {} from \'{}\''.format(own_metric, bin_in, bin_name, bin_name + '_mid', bin_name + '_width', selected_run)
        own_error_df = pd.read_sql_query(query, self.con)
        own_error_df.rename(columns={own_metric: metric}, inplace=True)
        if selected_comparison == 'IceCube':
            opponent_metric = 'opponent_' + metric + '_error'
            selected_comparison = selected_run
        else:
            opponent_metric = own_metric
        query = 'SELECT {}, {}, {}, {}, {} from \'{}\''.format(opponent_metric, bin_in, bin_name, bin_name + '_mid', bin_name + '_width', selected_comparison)
        opponent_error_df = pd.read_sql_query(query, self.con)
        opponent_error_df.rename(columns={opponent_metric: metric}, inplace=True)
        return own_error_df, opponent_error_df

    def drop_table(self, table_name):
        query = 'drop table if exists \'{}\';'.format(table_name)
        self.cursor.executescript(query)
