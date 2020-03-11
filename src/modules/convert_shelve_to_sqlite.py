import shelve
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import json

SEQ_STRING_KEYS = [
    'dom_key'
]
SEQ_FLOAT_KEYS = [
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_charge',
]
SEQ_INT_KEYS = [
    'dom_time',
    'dom_lc',
    'dom_atwd',
    'dom_fadc',
    'dom_pulse_width'
]
SCALAR_FLOAT_KEYS = [
    'dom_timelength_fwhm',
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z',
    'true_primary_speed',
    'true_primary_time',
    'true_primary_energy',
    'linefit_direction_x',
    'linefit_direction_y',
    'linefit_direction_z',
    'linefit_point_on_line_x',
    'linefit_point_on_line_y',
    'linefit_point_on_line_z',
    'toi_direction_x',
    'toi_direction_y',
    'toi_direction_z',
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'toi_evalratio',
    'retro_crs_prefit_x',
    'retro_crs_prefit_y',
    'retro_crs_prefit_z',
    'retro_crs_prefit_azimuth',
    'retro_crs_prefit_zenith',
    'retro_crs_prefit_time',
    'retro_crs_prefit_energy'
]
SCALAR_INT_KEYS = [
    'dom_n_hit_multiple_doms',
]
SCALAR_STRING_KEYS = [
    'secondary_track_length'
]
MASK_KEYS = [
    'SplitInIcePulses',
    'SRTInIcePulses'
]
META_KEYS = [
    'file',
    'index',
    'particle_code',
    'level'
]

SHELVE_PATH = Path().home().joinpath('files/icecube/oscnext-genie-level5-v01-01-pass2/shelve')
SQLITE_PATH = Path().home().joinpath('files/icecube/oscnext-genie-level5-v01-01-pass2/sqlite')

CONN = sqlite3.connect(str(SQLITE_PATH) + '/train_set_sqlite.db')
C = CONN.cursor()

CREATE_SEQ_TABLE_QUERY = '''
CREATE TABLE sequential (
    row INTEGER PRIMARY KEY,
    event_no INTEGER NOT NULL,
    pulse_no INTEGER NOT NULL,
    dom_key TEXT NOT NULL,
    dom_x REAL NOT NULL,
    dom_y REAL NOT NULL,
    dom_z REAL NOT NULL,
    dom_charge REAL NOT NULL,
    dom_time INTEGER NOT NULL,
    dom_lc INTEGER NOT NULL,
    dom_atwd INTEGER NOT NULL,
    dom_fadc INTEGER NOT NULL,
    dom_pulse_width INTEGER NOT NULL,
    SplitInIcePulses INTEGER NOT NULL,
    SRTInIcePulses INTEGER NOT NULL
);
'''
CREATE_SCALAR_TABLE_QUERY = '''
CREATE TABLE scalar (
    event_no INTEGER PRIMARY KEY,
    dom_timelength_fwhm REAL,
    true_primary_direction_x REAL,
    true_primary_direction_y REAL,
    true_primary_direction_z REAL,
    true_primary_position_x REAL,
    true_primary_position_y REAL,
    true_primary_position_z REAL,
    true_primary_speed REAL,
    true_primary_time REAL,
    true_primary_energy REAL,
    linefit_direction_x REAL,
    linefit_direction_y REAL,
    linefit_direction_z REAL,
    linefit_point_on_line_x REAL,
    linefit_point_on_line_y REAL,
    linefit_point_on_line_z REAL,
    toi_direction_x REAL,
    toi_direction_y REAL,
    toi_direction_z REAL,
    toi_point_on_line_x REAL,
    toi_point_on_line_y REAL,
    toi_point_on_line_z REAL,
    toi_evalratio REAL,
    retro_crs_prefit_x REAL,
    retro_crs_prefit_y REAL,
    retro_crs_prefit_z REAL,
    retro_crs_prefit_azimuth REAL,
    retro_crs_prefit_zenith REAL,
    retro_crs_prefit_time REAL,
    retro_crs_prefit_energy REAL,
    dom_n_hit_multiple_doms INTEGER
);
'''
CREATE_META_TABLE_QUERY = '''
CREATE TABLE meta (
    event_no INTEGER PRIMARY KEY,
    file TEXT,
    idx INTEGER,
    particle_code INTEGER,
    level INTEGER,
    split_in_ice_pulses_event_length INTEGER,
    srt_in_ice_pulses_event_length INTEGER
);
'''

# C.execute('''CREATE TABLE sequential
#              {}'''.format(tuple(['event_no', 'pulse_no'] + SEQ_STRING_KEYS + SEQ_FLOAT_KEYS + SEQ_INT_KEYS + MASK_KEYS)))
# C.execute('''CREATE TABLE scalar
#              {}'''.format(tuple(['event_no'] + SCALAR_FLOAT_KEYS + SCALAR_INT_KEYS + SCALAR_STRING_KEYS)))
# C.execute('''CREATE TABLE meta
#              {}'''.format(tuple(['event_no'] + META_KEYS + ['split_in_ice_pulses_event_length', 'srt_in_ice_pulses_event_length'])))
C.execute(CREATE_SEQ_TABLE_QUERY)
C.execute(CREATE_SCALAR_TABLE_QUERY)
C.execute(CREATE_META_TABLE_QUERY)
# C.execute('''CREATE TABLE blah (id, data)''')

row = 0

with shelve.open(str(SHELVE_PATH) + '/train_set', 'r') as f:
    print('{}: starting conversion'.format(datetime.now().time().strftime('%H:%M:%S')))
    keys = list(f.keys())
    for i, event_no in enumerate(keys):
        # event = f[event_no]
        # for key in event:
        #     for attribute in event[key]:
        #         print(type(event[key][attribute]))
        #         if isinstance(event[key][attribute], np.ndarray):
        #             event[key][attribute] = event[key][attribute].tolist()
        # C.execute('insert into countries values (?, ?)', [event_no, json.dumps(f[event_no])])
        event_raw = f[event_no]['raw']
        event_length = len(event_raw['dom_x'])
        event_masks = f[event_no]['masks']
        for key in SEQ_STRING_KEYS:
            seq_strings = event_raw[key]
        seq_floats = np.zeros((event_length, len(SEQ_FLOAT_KEYS)))
        for j, key in enumerate(SEQ_FLOAT_KEYS):
            seq_floats[:, j] = event_raw[key]
        seq_ints = np.zeros((event_length, len(SEQ_INT_KEYS)))
        for j, key in enumerate(SEQ_INT_KEYS):
            seq_ints[:, j] = event_raw[key]
        masks = np.zeros((event_length, 2))
        for j, key in enumerate(MASK_KEYS):
            for k in range(event_length):
                masks[k, j] = 1 if k in event_masks[key] else 0
        for j in range(seq_floats.shape[0]):
            tuple0 = tuple([int(row)], )
            tuple1 = tuple([int(event_no)],)
            tuple2 = tuple([j])
            tuple3 = tuple([seq_strings[j]],)
            tuple4 = tuple(seq_floats[j, :])
            tuple5 = tuple(seq_ints[j, :].astype(int))
            tuple6 = tuple([int(masks[j, 0])])
            tuple7 = tuple([int(masks[j, 1])])
            final_tuple = tuple0 + tuple1 + tuple2 + tuple3 + tuple4 + tuple5 + tuple6 + tuple7
            C.execute('INSERT INTO sequential VALUES {}'.format(final_tuple))
            row += 1
        scalar_floats = np.zeros((1, len(SCALAR_FLOAT_KEYS)))
        for j, key in enumerate(SCALAR_FLOAT_KEYS):
            scalar_floats[:, j] = event_raw[key]
        scalar_ints = np.zeros((1, len(SCALAR_INT_KEYS)))
        for j, key in enumerate(SCALAR_INT_KEYS):
            scalar_ints[:, j] = event_raw[key]
        for j, key in enumerate(SCALAR_STRING_KEYS):
            scalar_strings = str(event_raw[key])
        tuple0 = tuple([int(event_no)])
        tuple1 = tuple(scalar_floats[0, :], )
        tuple2 = tuple(scalar_ints[0, :].astype(int), )
        final_tuple = tuple0 + tuple1 + tuple2
        C.execute('INSERT INTO scalar VALUES {}'.format(final_tuple))
        event_meta = f[event_no]['meta']
        meta = []
        split_in_ice_pulses_event_length = event_length
        srt_in_ice_pulses_event_length = len(masks[:, 1][masks[:, 1] > 0])
        for j, key in enumerate(META_KEYS):
            meta.append(event_meta[key])
        tuple0 = tuple([int(event_no)])
        tuple1 = tuple(meta)
        tuple2 = tuple([int(split_in_ice_pulses_event_length)])
        tuple3 = tuple([int(srt_in_ice_pulses_event_length)])
        final_tuple = tuple0 + tuple1 + tuple2 + tuple3
        C.execute('INSERT INTO meta VALUES {}'.format(final_tuple))
        if i % 10000 == 0 and i > 0:
            CONN.commit()
            print('{}: handled {} events'.format(datetime.now().time().strftime('%H:%M:%S'), i))
        # if i == 1000:
        #     break
C.execute('''CREATE INDEX sequential_idx ON sequential(event_no)''')
C.execute('''CREATE UNIQUE INDEX scalar_idx ON scalar(event_no)''')
C.execute('''CREATE UNIQUE INDEX meta_idx ON meta(event_no)''')
CONN.commit()
CONN.close()
print('{}: Done!'.format(datetime.now().time().strftime('%H:%M:%S')))
