from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
import pickle
from multiprocessing import Pool, cpu_count
from itertools import product

sql_path = Path('/mnt/c/Users/MadsEhrhorn/Downloads/')
sql_file = sql_path.joinpath('train_set.db')
masks_path = Path().home().joinpath('masks')
masks_file = masks_path.joinpath('muon_neutrino.pickle')
with open(str(masks_file), 'rb') as f:
    mask = pickle.load(f)


def get_training_batch(event_nos):
    start = datetime.now()
    con = sqlite3.connect(sql_file)
    cur = con.cursor()
    # query = 'SELECT * FROM sequential WHERE event IN ({})'.format(', '.join(['?' for _ in event_nos]))
    query = 'SELECT * FROM sequential WHERE event IN ({seq})'.format(seq=','.join(['?'] * len(event_nos)))
    cur.execute(query, event_nos)
    fetched_sequential = cur.fetchall()
    # fetched_sequential = pd.read_sql(query, con, params=[*event_nos])
    # query = 'SELECT * FROM scalar WHERE event IN ({})'.format(', '.join(['?' for _ in event_nos]))
    # fetched_scalar = pd.read_sql(query, con, params=[*event_nos])
    con.close()
    end = datetime.now()
    delta_time = (end - start).total_seconds()
    return delta_time


if __name__ == '__main__':
    batch_size = 64
    total_time = []
    total_events = []
    event_predictions = []
    events = []
    pool = Pool(processes=(4))

    for i in range(0, 7280448, batch_size):
        events.append(mask[i:i + batch_size])

    events = events[0:1024]

    start = datetime.now()
    pool.map(get_training_batch, events)
    pool.close()
    pool.join()
    end = datetime.now()
    delta_time = round((end - start).total_seconds(), 2)

no_of_events = len(events) * batch_size
batches_per_second = int(no_of_events / delta_time)
print('Grabbed {} events, {} events per second. Script took {} seconds'.format(no_of_events, batches_per_second, delta_time))
