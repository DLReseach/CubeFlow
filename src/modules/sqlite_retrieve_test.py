import sqlite3
from pathlib import Path
from datetime import datetime

SQL_DB_PATH = Path().home().joinpath('files/icecube/oscnext-genie-level5-v01-01-pass2/sqlite/test_set_indexed_2.db')

FEATURES = ('dom_x',
    'dom_y',
    'dom_z',
    'dom_charge',
    'dom_time'
)
TARGETS = (
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_energy',
    'true_primary_time',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z'
)

FEATURES = str(FEATURES).replace('(', '').replace(')', '')
print(FEATURES)
TARGETS = str(TARGETS).replace('(', '').replace(')', '')

conn = sqlite3.connect(str(SQL_DB_PATH))
cur = conn.cursor()
cur.execute('SELECT DISTINCT(event_no) FROM sequential')
start = datetime.now()
rows = cur.fetchall()
end = datetime.now()
elapsed_time = (end - start).total_seconds()
print('Fetching event_nos took {} seconds'.format(elapsed_time))
conn.close()

for j in range(0, 2048, 64):
    conn = sqlite3.connect(str(SQL_DB_PATH))
    cur = conn.cursor()
    start = datetime.now()
    events = rows[j: j + 128]
    blah = []
    for event in events:
        blah.append(event[0])
    blah = str(blah).replace('[', '').replace(']', '')
    cur.execute('SELECT dom_x, dom_y, dom_z, dom_charge, dom_time FROM sequential WHERE event_no IN ({})'.format(blah))
    event_sequential = cur.fetchall()
    print(len(event_sequential))
    # cur.execute('SELECT {} FROM scalar WHERE event_no IN ({})'.format(TARGETS, blah))
    # event_scalar = cur.fetchall()
    end = datetime.now()
    elapsed_time = (end - start).total_seconds()
    conn.close()
    print('Fetching one event took {} seconds'.format(elapsed_time))

# print(event_sequential)
# print(event_scalar)
