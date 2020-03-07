from pathlib import Path
import shelve
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from src.modules.database_schema import Sequential, Scalar, Meta

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
    'retro_crs_prefit_energy',
    'secondary_track_length'
]
SCALAR_INT_KEYS = [
    'dom_n_hit_multiple_doms',
]
MASK_KEYS = [
    'SplitInIcePulses',
    'SRTInIcePulses'
]
META_STRING_KEYS = [
    'file'
]
META_INT_KEYS = [
    'idx',
    'particle_code',
    'level'
]

SHELVE_PATH = Path().home().joinpath('files/icecube/oscnext-genie-level5-v01-01-pass2/shelve')
SQLITE_PATH = Path().home().joinpath('files/icecube/oscnext-genie-level5-v01-01-pass2/sqlite')
SQLITE_FILE = SQLITE_PATH.joinpath('test_set.db')

Base = declarative_base()
engine = create_engine('sqlite:///' + str(SQLITE_FILE), echo=False)
connection = engine.connect()
Session = sessionmaker(bind=connection)
session = Session()
Base.metadata.create_all(engine)

row = 0

with shelve.open(str(SHELVE_PATH) + '/test_set', 'r') as f:
    print('{}: starting conversion'.format(datetime.now().time().strftime('%H:%M:%S')))
    keys = list(f.keys())
    for i, event_no in enumerate(keys):
        event_raw = f[event_no]['raw']
        event_meta = f[event_no]['meta']
        event_masks = f[event_no]['masks']
        event_length = len(event_raw['dom_x'])
        masks = np.zeros((event_length, 2))
        for j in range(event_length):
            sequential_dict = {}
            for key in SEQ_STRING_KEYS:
                sequential_dict[key] = str(event_raw[key][j])
            for key in SEQ_FLOAT_KEYS:
                sequential_dict[key] = float(event_raw[key][j])
            for key in SEQ_INT_KEYS:
                sequential_dict[key] = int(event_raw[key][j])
            for key in MASK_KEYS:
                sequential_dict[key] = int(1) if j in event_masks[key] else int(0)
            sequential_dict['event'] = int(event_no)
            sequential_dict['row'] = int(row)
            sequential_dict['pulse'] = int(j)
            row += 1
            session.execute(Sequential.__table__.insert(),
                [
                    sequential_dict
                ]
            )
        scalar_dict = {}
        for key in SCALAR_FLOAT_KEYS:
            scalar_dict[key] = float(event_raw[key])
        for key in SCALAR_INT_KEYS:
            scalar_dict[key] = int(event_raw[key])
        scalar_dict['event'] = int(event_no)
        session.execute(Scalar.__table__.insert(),
            [
                scalar_dict
            ]
        )
        for j, key in enumerate(MASK_KEYS):
            for k in range(event_length):
                masks[k, j] = 1 if k in event_masks[key] else 0
        meta_dict = {}
        for key in META_STRING_KEYS:
            meta_dict[key] = str(event_meta[key])
        for key in META_INT_KEYS:
            if key == 'idx':
                meta_dict[key] = int(event_meta['index'])
            else:
                meta_dict[key] = int(event_meta[key])
        meta_dict['event'] = int(event_no)
        meta_dict['split_in_ice_pulses_event_length'] = event_length
        meta_dict['srt_in_ice_pulses_event_length'] = len(masks[:, 1][masks[:, 1] > 0])
        session.execute(Meta.__table__.insert(),
            [
                meta_dict
            ]
        )
        if i % 10000 == 0 and i > 0:
            session.commit()
            print('{}: handled {} events'.format(datetime.now().time().strftime('%H:%M:%S'), i))
            break
session.commit()
seq_index = Index('sequential_idx', Sequential.event)
scalar_index = Index('scalar_idx', Scalar.event, unique=True)
meta_index = Index('meta_idx', Meta.event, unique=True)
seq_index.create(bind=engine)
scalar_index.create(bind=engine)
meta_index.create(bind=engine)
session.commit()
print('{}: Done!'.format(datetime.now().time().strftime('%H:%M:%S')))
