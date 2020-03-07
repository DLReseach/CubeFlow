from pathlib import Path
import shelve
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Sequential(Base):
    __tablename__ = 'sequential'
    row = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    event = Column(Integer)
    pulse = Column(Integer, nullable=False)
    dom_key = Column(String, nullable=False)
    dom_x = Column(Float, nullable=False)
    dom_y = Column(Float, nullable=False)
    dom_z = Column(Float, nullable=False)
    dom_charge = Column(Float, nullable=False)
    dom_time = Column(Integer, nullable=False)
    dom_lc = Column(Integer, nullable=False)
    dom_atwd = Column(Integer, nullable=False)
    dom_fadc = Column(Integer, nullable=False)
    dom_pulse_width = Column(Integer, nullable=False)
    SplitInIcePulses = Column(Integer, nullable=False)
    SRTInIcePulses = Column(Integer, nullable=False)


class Scalar(Base):
    __tablename__ = 'scalar'
    event = Column(Integer, primary_key=True, nullable=False)
    dom_timelength_fwhm = Column(Float, nullable=False)
    true_primary_direction_x = Column(Float, nullable=False)
    true_primary_direction_y = Column(Float, nullable=False)
    true_primary_direction_z = Column(Float, nullable=False)
    true_primary_position_x = Column(Float, nullable=False)
    true_primary_position_y = Column(Float, nullable=False)
    true_primary_position_z = Column(Float, nullable=False)
    true_primary_speed = Column(Float, nullable=False)
    true_primary_time = Column(Float, nullable=False)
    true_primary_energy = Column(Float, nullable=False)
    linefit_direction_x = Column(Float, nullable=False)
    linefit_direction_y = Column(Float, nullable=False)
    linefit_direction_z = Column(Float, nullable=False)
    linefit_point_on_line_x = Column(Float, nullable=False)
    linefit_point_on_line_y = Column(Float, nullable=False)
    linefit_point_on_line_z = Column(Float, nullable=False)
    toi_direction_x = Column(Float, nullable=False)
    toi_direction_y = Column(Float, nullable=False)
    toi_direction_z = Column(Float, nullable=False)
    toi_point_on_line_x = Column(Float, nullable=False)
    toi_point_on_line_y = Column(Float, nullable=False)
    toi_point_on_line_z = Column(Float, nullable=False)
    toi_evalratio = Column(Float, nullable=False)
    retro_crs_prefit_x = Column(Float, nullable=False)
    retro_crs_prefit_y = Column(Float, nullable=False)
    retro_crs_prefit_z = Column(Float, nullable=False)
    retro_crs_prefit_azimuth = Column(Float, nullable=False)
    retro_crs_prefit_zenith = Column(Float, nullable=False)
    retro_crs_prefit_time = Column(Float, nullable=False)
    retro_crs_prefit_energy = Column(Float, nullable=False)
    dom_n_hit_multiple_doms = Column(Integer, nullable=False)
    secondary_track_length = Column(Float, nullable=True)


class Meta(Base):
    __tablename__ = 'meta'
    event = Column(Integer, primary_key=True, nullable=False)
    file = Column(String, nullable=False)
    idx = Column(Integer, nullable=False)
    particle_code = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False)
    split_in_ice_pulses_event_length = Column(Integer, nullable=False)
    srt_in_ice_pulses_event_length = Column(Integer, nullable=False)


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
SQLITE_FILE = SQLITE_PATH.joinpath('train_set.db')

engine = create_engine('sqlite:///' + str(SQLITE_FILE), echo=False)
connection = engine.connect()
Session = sessionmaker(bind=connection)
session = Session()
Base.metadata.create_all(engine)

total_event_time = []

with shelve.open(str(SHELVE_PATH) + '/train_set', 'r') as f:
    print('{}: starting conversion'.format(datetime.now().time().strftime('%H:%M:%S')))
    keys = list(f.keys())
    start = datetime.now()
    for i, event_no in enumerate(keys):
        start_event = datetime.now()
        event_raw = f[event_no]['raw']
        event_meta = f[event_no]['meta']
        event_masks = f[event_no]['masks']
        event_length = len(event_raw['dom_x'])
        masks = np.zeros((event_length, 2))
        sequential_insert_list = []
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
            sequential_dict['pulse'] = int(j)
            sequential_insert_list.append(sequential_dict)
        session.bulk_insert_mappings(Sequential, sequential_insert_list)
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
        end_event = datetime.now()
        delta_event = (end_event - start_event).total_seconds()
        total_event_time.append(delta_event)
        if i % 10000 == 0 and i > 0:
            session.commit()
            mean = sum(total_event_time) / len(total_event_time)
            total_event_time = []
            end = datetime.now()
            time_delta = round((end - start).total_seconds(), 3)
            print('{}: handled {} events, took {} seconds; each event took {} seconds on average'.format(datetime.now().time().strftime('%H:%M:%S'), i, time_delta, mean))
            start = datetime.now()
        # if i == 100:
        #     break
session.commit()
seq_index = Index('sequential_idx', Sequential.event)
scalar_index = Index('scalar_idx', Scalar.event, unique=True)
meta_index = Index('meta_idx', Meta.event, unique=True)
seq_index.create(bind=engine)
scalar_index.create(bind=engine)
meta_index.create(bind=engine)
session.commit()
print('{}: Done!'.format(datetime.now().time().strftime('%H:%M:%S')))
