import torch
import sqlite3
import numpy as np


class SqlDataset(torch.utils.data.Dataset):
      '''Dataset fetching from SQLite database.'''
      def __init__(self, mask, sql_file, features, targets, max_doms, cleaning, batch_size):
            self.mask = mask
            self.sql_file = sql_file
            self.features = features
            self.targets = targets
            self.max_doms = max_doms
            self.cleaning = cleaning
            self.batch_size = batch_size
            self.floor_events = int(len(mask) // batch_size * batch_size)
            self.no_of_batches = int(len(mask) // batch_size)
            # Shuffle events at init
            self._on_epoch_start()

      def __len__(self):
            '''Standard __len__.'''
            return len(self.events)

      def _on_epoch_start(self):
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
            X, y, z = self._coerce_batch(batch_events)
            # Shuffle events list at last index
            if index == self.no_of_batches:
                  self._on_epoch_start()
            return X, y, z

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
            query = 'SELECT {features} FROM sequential WHERE event IN ({events})'.format(
                  features=', '.join(self.features + [self.cleaning, 'pulse']),
                  events=', '.join(['?'] * len(events))
            )
            cur.execute(query, events)
            fetched_sequential = cur.fetchall()
            # Write query for scalar table and fetch all matching rows
            query = 'SELECT {targets} FROM scalar WHERE event IN ({events})'.format(
                  targets=', '.join(self.targets),    
                  events=', '.join(['?'] * len(events))
            )
            cur.execute(query, events)
            fetched_scalar = cur.fetchall()
            # Write query for meta table and fetch all matching rows
            query = 'SELECT {cleaning} FROM meta WHERE event IN ({events})'.format(
                  cleaning=self.cleaning,
                  events=', '.join(['?'] * len(events))
            )
            cur.execute(query, events)
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
            max_length = max(lengths)[0]
            # Preallocation of arrays
            X = np.zeros((len(events), max_length, len(self.features)))
            y = np.zeros((len(events), len(self.targets)))
            # Array for testing (saving to csv)
            z = np.zeros((len(events), max_length, 3))
            # Set counters for coercion
            i = 0
            j = 0
            for pulse in fetched_sequential:
                  # On the first pulse in a sequence
                  if pulse[-1] == 0:
                        event_length = lengths[i][0]
                        # sqlite3 returns events with event number sorted
                        z[i, :, 0] = sorted(events)[i]
                        z[i, :, 1] = np.arange(0, max_length)
                        # Calculate insertion point in final array dimension 1; for start/end padding
                        insert_point = int((max_length - event_length) // 2)
                        i += 1
                        # Row counter
                        j = 0
                        # Pulse counter
                        k = 0
                  if pulse[-2] == 1:
                        # Insert features in the 'middle' of the array
                        X[i - 1, insert_point + j, :] = pulse[0:len(self.features)]
                        # Pulse number
                        z[i - 1, insert_point + j, 2] = k
                        j += 1
                  k += 1
            for i, target in enumerate(fetched_scalar):
                  y[i, :] = target
            return X, y, z
