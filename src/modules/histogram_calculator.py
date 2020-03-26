import numpy as np
import pandas as pd
import sqlite3
import pickle
from pathlib import Path


class HistogramCalculator:
    def __init__(self, masks_name, run_name, dirs):
        self.masks_name = masks_name
        self.run_name = run_name
        self.dirs = dirs

        self.errors_db_path = dirs['dbs'].joinpath(self.masks_name).joinpath('errors.db')
        self.predictions_db_path = dirs['dbs'].joinpath(self.masks_name).joinpath('predictions.db')
        pickle_path = dirs['dbs'].joinpath(self.masks_name)
        self.pickle_file = pickle_path.joinpath('histograms.pkl')

        if not self.pickle_file.is_file():
            self.create_new_dictionary()
            self.calculate_histograms('retro_crs_prefit')
        else:
            with open(self.pickle_file, 'rb') as f:
                self.dictionary = pickle.load(f)
        self.non_na_events = []
        for ibin in self.dictionary['meta']['events']:
            self.non_na_events.extend(self.dictionary['meta']['events'][ibin])
        self.calculate_histograms(run_name)
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.dictionary, f)

    def create_new_dictionary(self):
        self.dictionary = {}
        self.dictionary['runs'] = {}
        self.dictionary['meta'] = {}
        self.dictionary['meta']['events'] = {}
        self.dictionary['meta']['1d_histogram'] = {}
        self.dictionary['meta']['2d_histogram'] = {}
        self.dictionary['meta']['1d_histogram']['bins'] = []  
        self.dictionary['meta']['1d_histogram']['bin_mids'] = []
        self.dictionary['meta']['1d_histogram']['bin_widths'] = []
        self.dictionary['meta']['1d_histogram']['counts'] = []    
        self.dictionary['meta']['2d_histogram']['x_edges'] = []
        self.dictionary['meta']['2d_histogram']['y_edges'] = []

    def create_run_dictionary(self, run_name, errors):
        self.dictionary['runs'][run_name] = {}
        for error in errors:
            self.dictionary['runs'][run_name][error] = {}
            self.dictionary['runs'][run_name][error]['1d_histogram'] = {}
            self.dictionary['runs'][run_name][error]['2d_histogram'] = {}
            self.dictionary['runs'][run_name][error]['2d_histogram']['percentiles'] = {}
            self.dictionary['runs'][run_name][error]['2d_histogram']['counts'] = []
            self.dictionary['runs'][run_name][error]['error_histograms'] = {}

    def calculate_1d_histogram(self, df):
        df['energy_binned'] = pd.cut(
            df['true_primary_energy'],
            18
        )
        bin_mids = [ibin.mid for ibin in np.sort(df['energy_binned'].unique())]
        bin_widths = [
            ibin.length / 2 for ibin in np.sort(df['energy_binned'].unique())
        ]
        bins = np.sort(df['energy_binned'].unique()).astype(str)
        counts = df.groupby(['energy_binned']).count()['true_primary_energy'].values
        return bins, bin_mids, bin_widths, counts

    def calculate_error_histograms(self, df, run_name, error):
        bins = np.sort(df['energy_binned'].unique())
        self.dictionary['runs'][run_name][error]['1d_histogram']['performance'] = []
        for i, ibin in enumerate(bins):
            self.dictionary['runs'][run_name][error]['error_histograms'][i] = {}
            hist, bin_edges = np.histogram(
                df[df['energy_binned'] == ibin][error].values,
                bins=100
            )
            low = np.nanpercentile(df[df['energy_binned'] == ibin][error].values, 16)
            mid = np.nanpercentile(df[df['energy_binned'] == ibin][error].values, 50)
            high = np.nanpercentile(df[df['energy_binned'] == ibin][error].values, 84)
            self.dictionary['runs'][run_name][error]['error_histograms'][i]['counts'] = hist
            self.dictionary['runs'][run_name][error]['error_histograms'][i]['x_edges'] = bin_edges
            self.dictionary['runs'][run_name][error]['error_histograms'][i]['percentiles'] = [low, mid, high]
            self.dictionary['runs'][run_name][error]['1d_histogram']['performance'].append((high - low) / 2)

    def calculate_histograms(self, run_name):
        errors_query = 'SELECT * FROM \"{table}\"'.format(
            table=run_name
        )
        with sqlite3.connect(self.errors_db_path) as con:
            errors = pd.read_sql_query(errors_query, con=con)
        truths_query = 'SELECT event_no, energy FROM truth'
        with sqlite3.connect(self.predictions_db_path) as con:
            truths = pd.read_sql_query(truths_query, con=con)
        merged_errors = errors.merge(truths, on='event_no', suffixes=('_x', '_y'))
        merged_errors.rename(columns={'energy_y': 'true_primary_energy'}, inplace=True)
        merged_errors.rename(columns={'energy_x': 'energy'}, inplace=True)

        if run_name == 'retro_crs_prefit':
            merged_errors.dropna(inplace=True)
        else:
            merged_errors = merged_errors[merged_errors['event_no'].isin(self.non_na_events)]
        merged_errors = merged_errors[merged_errors['true_primary_energy'] <= 3.0]
        bins, mids, widths, counts = self.calculate_1d_histogram(merged_errors)
        errors = [error for error in merged_errors.columns.values if error not in ['event_no', 'true_primary_energy', 'energy_binned']]
        self.create_run_dictionary(run_name, errors)
        for error in errors:
            self.calculate_error_histograms(
                merged_errors,
                run_name,
                error
            )
        if run_name == 'retro_crs_prefit':
            self.dictionary['meta']['1d_histogram']['bins'] = bins
            self.dictionary['meta']['1d_histogram']['bin_mids'] = mids
            self.dictionary['meta']['1d_histogram']['bin_widths'] = widths
            self.dictionary['meta']['1d_histogram']['counts'] = counts
            for i, ibin in enumerate(np.sort(merged_errors['energy_binned'].unique())):
                self.dictionary['meta']['events'][i] = merged_errors[merged_errors['energy_binned'] == ibin]['event_no'].values
