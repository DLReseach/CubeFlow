import torch
import sqlite3
import numpy as np
from pathlib import Path


class Dataloader(torch.utils.data.Dataset):
    '''Dataset fetching from SQLite database.'''
    def __init__(self, mask, config, sql_file, test):
        self.mask = mask
        self.config = config
        self.sql_file = sql_file
        self.test = test

        if self.test:
            self.batch_size = config['val_batch_size']
        else:
            self.batch_size = config['batch_size']

        self.floor_events = int(len(mask) // self.batch_size * self.batch_size)
        self.no_of_batches = int(len(mask) // self.batch_size)

        # Shuffle events at init
        self.on_epoch_start()

    def __len__(self):
        '''Standard __len__.'''
        return len(self.events)

    def on_epoch_start(self):
        '''Shuffle events at epoch start.'''
        # Shuffle events list
        np.random.shuffle(self.mask)
        self.events = []
        # Batch events in batch sizes
        for i in range(0, self.floor_events, self.batch_size):
            self.events.append(self.mask[i:i + self.batch_size])

    def __getitem__(self, index):
        '''Standard __getitem__.

        Shuffles events at last batch index.

        Args:
              index (int): Batch index.

        Returns:
            tuple: tuple containing:
                X (numpy.ndarray): Coerced array containing batched padded features.
                y (numpy.ndarray): Batched targets.
                z (numpy.ndarray): Event numbers and rows used for testing.
        '''
        # Retrieve batch from events list
        batch_events = self.events[index]
        X, y, events = self._coerce_batch(batch_events)
        return X, y, events

    def _get_from_sql(self, events):
        '''Retrieve events from a SQLite database.

        Args:
              events (list): List of events to retrieve.

        Returns:
              tuple: tuple containing:
                fetched_sequential (list): All rows fetched from sequential table.
                fetched_scalar (list): All rows fetched from scalar table.
                lengths (list): Length of each fetched event.
        '''
        # Connect to database and set cursor
        con = sqlite3.connect(self.sql_file)
        cur = con.cursor()
        # Write query for sequential table and fetch all matching rows
        query = 'SELECT {} FROM sequential WHERE event IN ({})'.format(
            ', '.join(self.config['features'] + [self.config['cleaning'], 'pulse_no']),
            ', '.join(str(event) for event in events)
        )
        cur.execute(query)
        fetched_sequential = cur.fetchall()
        # Write query for scalar table and fetch all matching rows
        query = 'SELECT {} FROM scalar WHERE event_no IN ({})'.format(
            ', '.join(self.config['targets']),
            ', '.join(str(event) for event in events)
        )
        cur.execute(query)
        fetched_scalar = cur.fetchall()
        # Write query for meta table and fetch all matching rows
        query = 'SELECT {} FROM meta WHERE event_no IN ({})'.format(
            self.config['cleaning_length'],
            ', '.join(str(event) for event in events)
        )
        cur.execute(query)
        lengths = cur.fetchall()
        # Close database connection
        con.close()
        return fetched_sequential, fetched_scalar, lengths

    def _coerce_batch(self, events):
        '''Retrieve events from a SQLite database and coerce + pad them.

        Args:
              events (list): List of events to retrieve.

        Returns:
            tuple: tuple containing:
                X (numpy.ndarray): Coerced array containing batched padded features.
                y (numpy.ndarray): Batched targets.
                z (numpy.ndarray): Event numbers and rows used for testing.
        '''
        # Get the rows from the database
        fetched_sequential, fetched_scalar, lengths = self._get_from_sql(events)
        # Length of longest event, used for padding
        # max_length = max(lengths)[0]
        max_length = self.config['max_doms']
        # Preallocation of arrays
        X = np.zeros((len(events), max_length, len(self.config['features'])))
        y = np.zeros((len(events), len(self.config['targets'])))
        # sqlite3 returns events with event number sorted
        events = sorted(events)
        # We can use torch convenience functions on Numpy arrays, not lists
        events = np.array(events)
        # Set counters for coercion
        i = 0
        j = 0
        for pulse in fetched_sequential:
            # On the first pulse in a sequence
            if pulse[-1] == 0:
                event_length = lengths[i][0]
                # Calculate insertion point in final array dimension 1; for start/end padding
                insert_point = int((max_length - event_length) // 2)
                i += 1
                # Row counter
                j = 0
            if pulse[-2] == 1:
                # Insert features in the 'middle' of the array
                X[i - 1, insert_point + j, :] = pulse[0:len(self.config['features'])]
                j += 1
        for i, target in enumerate(fetched_scalar):
            y[i, :] = target
        # Transpose so (batch, channels, pulses)
        X = np.transpose(X, axes=[0, 2, 1])
        return X, y, events
