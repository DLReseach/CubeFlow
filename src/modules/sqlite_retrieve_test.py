from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, Index, select
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Sequential(Base):
    __tablename__ = 'sequential'
    event_no = Column(Integer, primary_key=True)
    pulse_no = Column(Integer, nullable=True)
    dom_key = Column(String, nullable=True)
    dom_x = Column(Float, nullable=True)
    dom_y = Column(Float, nullable=True)
    dom_z = Column(Float, nullable=True)
    dom_charge = Column(Float, nullable=True)
    dom_time = Column(Integer, nullable=True)
    dom_lc = Column(Integer, nullable=True)
    dom_atwd = Column(Integer, nullable=True)
    dom_fadc = Column(Integer, nullable=True)
    dom_pulse_width = Column(Integer, nullable=True)
    SplitInIcePulses = Column(Integer, nullable=True)
    SRTInIcePulses = Column(Integer, nullable=True)


class Scalar(Base):
    __tablename__ = 'scalar'
    event_no = Column(Integer, primary_key=True, nullable=True)
    dom_timelength_fwhm = Column(Float, nullable=True)
    true_primary_direction_x = Column(Float, nullable=True)
    true_primary_direction_y = Column(Float, nullable=True)
    true_primary_direction_z = Column(Float, nullable=True)
    true_primary_position_x = Column(Float, nullable=True)
    true_primary_position_y = Column(Float, nullable=True)
    true_primary_position_z = Column(Float, nullable=True)
    true_primary_speed = Column(Float, nullable=True)
    true_primary_time = Column(Float, nullable=True)
    true_primary_energy = Column(Float, nullable=True)
    linefit_direction_x = Column(Float, nullable=True)
    linefit_direction_y = Column(Float, nullable=True)
    linefit_direction_z = Column(Float, nullable=True)
    linefit_point_on_line_x = Column(Float, nullable=True)
    linefit_point_on_line_y = Column(Float, nullable=True)
    linefit_point_on_line_z = Column(Float, nullable=True)
    toi_direction_x = Column(Float, nullable=True)
    toi_direction_y = Column(Float, nullable=True)
    toi_direction_z = Column(Float, nullable=True)
    toi_point_on_line_x = Column(Float, nullable=True)
    toi_point_on_line_y = Column(Float, nullable=True)
    toi_point_on_line_z = Column(Float, nullable=True)
    toi_evalratio = Column(Float, nullable=True)
    retro_crs_prefit_x = Column(Float, nullable=True)
    retro_crs_prefit_y = Column(Float, nullable=True)
    retro_crs_prefit_z = Column(Float, nullable=True)
    retro_crs_prefit_azimuth = Column(Float, nullable=True)
    retro_crs_prefit_zenith = Column(Float, nullable=True)
    retro_crs_prefit_time = Column(Float, nullable=True)
    retro_crs_prefit_energy = Column(Float, nullable=True)
    dom_n_hit_multiple_doms = Column(Integer, nullable=True)
    secondary_track_length = Column(Float, nullable=True)


class Meta(Base):
    __tablename__ = 'meta'
    event_no = Column(Integer, primary_key=True, nullable=True)
    file = Column(String, nullable=True)
    index = Column(Integer, nullable=True)
    particle_code = Column(Integer, nullable=True)
    level = Column(Integer, nullable=True)
    split_in_ice_pulses_event_length = Column(Integer, nullable=True)
    srt_in_ice_pulses_event_length = Column(Integer, nullable=True)


SQLITE_PATH = Path('/mnt/c/Users/MadsEhrhorn/Downloads/')
SQLITE_FILE = SQLITE_PATH.joinpath('test_set.db')

engine = create_engine('sqlite:///' + str(SQLITE_FILE), echo=False)
connection = engine.connect()
Session = sessionmaker(bind=connection)
session = Session()

event_nos = session.query(Meta.event_no).all()

for i in range(0, 1024, 64):
    events = ()
    temp_events = event_nos[i:i + 64]
    for j in range(len(temp_events)):
        events += temp_events[j]
    start = datetime.now()
    query = session.query(Sequential.dom_x, Sequential.event_no).filter(Sequential.event_no.in_(events))
    query_string = str(query.statement.compile(compile_kwargs={'literal_binds': True}))
    test_df = pd.read_sql(query_string, connection)
    results = query.all()
    end = datetime.now()
    time_delta = (end - start).total_seconds()
    print('Fetch {} event(s) took {} seconds'.format(len(events), time_delta))
print(test_df.head())
session.close()
