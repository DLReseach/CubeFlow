from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Sequential(Base):
    __tablename__ = 'sequential'
    row = Column(Integer, primary_key=True, nullable=False)
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
