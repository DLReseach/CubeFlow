import shelve
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

SEQ_KEYS = [
    'dom_key',
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_time',
    'dom_charge',
    'dom_lc',
    'dom_atwd',
    'dom_fadc',
    'dom_pulse_width'
]
SCALAR_KEYS = [
    'dom_timelength_fwhm',
    'dom_n_hit_multiple_doms',
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
    'retro_crs_prefit_energy',
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

CONN = sqlite3.connect('test_set.db')
C = CONN.cursor()

C.execute('''CREATE TABLE sequential
             {}'''.format(tuple(['event_no', 'pulse_no'] + SEQ_KEYS + MASK_KEYS)))
C.execute('''CREATE TABLE scalar
             {}'''.format(tuple(['event_no'] + SCALAR_KEYS)))
C.execute('''CREATE TABLE meta
             {}'''.format(tuple(['event_no'] + META_KEYS)))


with shelve.open(str(SHELVE_PATH) + '/test_set', 'r') as f:
    print('{}: starting conversion'.format(datetime.now().time().strftime('%H:%M:%S')))
    keys = list(f.keys())
    for i, event_no in enumerate(keys):
        event_raw = f[event_no]['raw']
        event_masks = f[event_no]['masks']
        pulses = np.zeros((len(event_raw['dom_x']), len(SEQ_KEYS) - 1))
        for j, key in enumerate(SEQ_KEYS):
            if key == 'dom_key':
                dom_keys = event_raw[key]
            else:
                pulses[:, j - 1] = event_raw[key]
        masks = np.zeros((len(event_raw['dom_x']), 2))
        for j, key in enumerate(MASK_KEYS):
            for k in range(len(event_raw['dom_x'])):
                masks[k, j] = 1 if k in event_masks[key] else 0
        for j in range(pulses.shape[0]):
            tuple0 = tuple([event_no],)
            tuple1 = tuple([j])
            tuple2 = tuple([dom_keys[j]],)
            tuple3 = tuple(pulses[j, :])
            tuple4 = tuple([int(masks[j, 0])])
            tuple5 = tuple([int(masks[j, 1])])
            final_tuple = tuple0 + tuple1 + tuple2 + tuple3 + tuple4 + tuple5
            C.execute("INSERT INTO sequential VALUES {}".format(final_tuple))
        scalars = np.zeros((1, len(SCALAR_KEYS) - 1))
        for j, key in enumerate(SCALAR_KEYS):
            if key == 'secondary_track_length':
                secondary_track_length = str(event_raw[key])
            else:
                scalars[:, j] = event_raw[key]
        tuple0 = tuple([event_no])
        tuple1 = tuple(scalars[0, :], )
        tuple2 = tuple([secondary_track_length])
        final_tuple = tuple0 + tuple1 + tuple2
        C.execute("INSERT INTO scalar VALUES {}".format(final_tuple))
        event_meta = f[event_no]['meta']
        meta = []
        for j, key in enumerate(META_KEYS):
            meta.append(event_meta[key])
        C.execute("INSERT INTO meta VALUES {}".format(tuple([event_no]) + tuple(meta)))
        if i % 500 == 0 and i > 0:
            print('{}: handled {} events'.format(datetime.now().time().strftime('%H:%M:%S'), i))

CONN.commit()
CONN.close()
print('{}: Done!'.format(datetime.now().time().strftime('%H:%M:%S')))
