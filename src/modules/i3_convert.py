"""Reading some atmopsheric muon sim files and grabbing some basic information.
How to run: `python i3_to_hdf5_calc.py`
Note that an IceTray environment _must_ be loaded.
Tom Stuttard
Ammendments by Mads Ehrhorn
"""
import os
import numpy as np
from pathlib import Path
from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube import tensor_of_inertia
from icecube.common_variables import hit_statistics
from icecube.common_variables import direct_hits
from icecube.common_variables import hit_multiplicity
from icecube.common_variables import time_characteristics
from argparse import ArgumentParser
import datetime
import shelve
import dbm.dumb as dumbdbm
import random

parser = ArgumentParser()
parser.add_argument(
    '-i',
    '--i3_type',
    dest='i3_type',
    help='What i3 type to download'
)
parser.add_argument(
    '-n',
    '--number_of_files_to_handle',
    dest='number_of_files_to_handle',
    help='How many files to handle (0 for all)'
)

args = parser.parse_args()
i3_type = args.i3_type
number_of_files_to_handle = int(args.number_of_files_to_handle)

if i3_type == 'MuonGun_Level2_139008':
    input_dir = Path(
        '/groups/icecube/stuttard/data/oscNext/pass2/muongun/level2/139008/'
    )
    save_dir = Path(
        '/groups/hep/ehrhorn/files/icecube/shelve/MuonGun_Level2_139008'
    )
    gcd_file = Path(
        '/groups/hep/ehrhorn/files/icecube/i3_files/Jason/'
        'GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3'
    )
elif i3_type == 'oscnext-genie-level5-v01-01-pass2':
    input_dir = Path(
        '/groups/icecube/stuttard/data/oscNext/pass2/genie/level5_v01.01/'
    )
    save_dir = Path(
        '/groups/hep/ehrhorn/files/icecube/oscnext-genie-level5-v01-01-pass2/shelve_test_2'
    )
    gcd_file = Path(
        '/groups/hep/ehrhorn/files/icecube/i3_files/Jason/'
        'GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3'
    )
else:
    raise Exception('Dunno the i3 type')

save_dir.mkdir(exist_ok=True, parents=True)

if __name__ == '__main__':
    i3_files = sorted([f for f in input_dir.glob('**/*.i3*') if f.is_file()])
    if number_of_files_to_handle != 0:
        i3_files = i3_files[0:number_of_files_to_handle]
    column_names = [
        'dom_key',
        'dom_x',
        'dom_y',
        'dom_z',
        'dom_time',
        'dom_charge',
        'dom_lc',
        'dom_atwd',
        'dom_fadc',
        'dom_pulse_width',
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
    mask_names = [
        'SplitInIcePulses',
        'SRTInIcePulses'
    ]
    meta_names = [
        'file',
        'index',
        'particle_code',
        'level'
    ]


    def do_things(frame):
        """Retrieve and calculate frames from an i3 file.
        Args:
            frame (i3 frame): i3 frame from i3 file.
        Returns:
            i3 object: i3 object with requested frames.
        """
        cond1 = frame['I3EventHeader'].sub_event_stream != 'InIceSplit'
        cond2 = 'LineFit' not in frame or 'ToI' not in frame
        cond3 = 'SRTInIcePulses' not in frame
        cond4 = not frame['L5_oscNext_bool']
        final_cond = cond1 or cond2 or cond3 or cond4
        if final_cond:
            return False
        else:
            linefit_result = frame['LineFit']
            linefit_fit_status = linefit_result.fit_status
            toi_result = frame['ToI']
            toi_fit_status = toi_result.fit_status
            try:
                retro_crs_prefit_result = frame[
                    'retro_crs_prefit__median__neutrino'
                ]
                retro_crs_prefit_status = frame[
                    'retro_crs_prefit__fit_status'
                ][0]
            except Exception:
                return False
            if linefit_fit_status != 0:
                return False
            elif toi_fit_status != 0:
                return False
            elif retro_crs_prefit_status != 0:
                return False

        hit_stats = frame['HitStatisticsValues']
        hit_mult = frame['HitMultiplicityValues']
        data['dom_n_hit_multiple_doms'].append(
            hit_mult.n_hit_doms - hit_mult.n_hit_doms_one_pulse
        )
        time_char = frame['TimeCharacteristicsValues']

        data['dom_timelength_fwhm'].append(time_char.timelength_fwhm)

        # TODO Could we maybe do this in a smarter, programmatic way?
        # Get the true muon particle
        # Note that the muon is really produced in the atmosphere, but for the
        # simulation it is generated at the surface of a cylinder
        # surrounding IceCube
        # <icecube.dataclasses.I3Particle>
        true_primary = dataclasses.get_most_energetic_primary(
            frame['I3MCTree']
        )
        true_muon = dataclasses.get_most_energetic_muon(frame['I3MCTree'])
        if true_muon == None:
            data['secondary_track_length'].append(np.nan)
        else:
            data['secondary_track_length'].append(true_muon.length)
        # Add the muon energy (defined at the generation cylinder) to the
        # data dicitonary
        data['true_primary_energy'].append(np.log10(true_primary.energy))

        # Direction of the muon
        # <icecube.dataclasses.I3Particle>
        true_primary_direction = true_primary.dir
        # Add the muon direction to the data dictionary
        data['true_primary_direction_x'].append(true_primary_direction.x)
        data['true_primary_direction_y'].append(true_primary_direction.y)
        data['true_primary_direction_z'].append(true_primary_direction.z)

        data['true_primary_time'].append(true_primary.time)
        data['true_primary_speed'].append(true_primary.speed)

        # Point on the generation cylinder at which the muon is produced
        # <icecube.dataclasses.I3Particle>
        true_primary_entry_position = true_primary.pos
        # Add the entry position to the data dictionary
        data['true_primary_position_x'].append(
            true_primary_entry_position.x
        )
        data['true_primary_position_y'].append(
            true_primary_entry_position.y
        )
        data['true_primary_position_z'].append(
            true_primary_entry_position.z
        )

        # Get the uncleaned pulses
        # <icecube.dataclasses.I3RecoPulseSeriesMap>
        uncleaned_pulses = frame['SplitInIcePulses'].apply(frame)
        cleaned_pulses = frame['SRTInIcePulses'].apply(frame)

        dom_geom = frame['I3Geometry'].omgeo

        # Create empty lists for holding the pulse information
        temp = np.empty((0, 11))
        dom_key_temp = []
        # dom_x_temp = np.empty(0)
        # dom_y_temp = np.empty(0)
        # dom_z_temp = np.empty(0)
        # dom_time_temp = np.empty(0)
        # dom_charge_temp = np.empty(0)
        # dom_lc_temp = np.empty(0)
        # dom_atwd_temp = np.empty(0)
        # dom_fadc_temp = np.empty(0)
        # dom_pulse_width_temp = np.empty(0)

        # Go through all pulses, get OM key and pair with time and charge info
        for entry in uncleaned_pulses.items():
            cleaned_time_list = []
            this_om_key = entry[0]
            if this_om_key in cleaned_pulses.keys():
                cleaned_temp = cleaned_pulses[this_om_key]
                for cleaned_entry in cleaned_temp:
                    cleaned_time_list.append(cleaned_entry.time)
            # This grabs you an object containing the geometry for this
            # particular OM
            this_om_geom = dom_geom[this_om_key]
            # This has x,y,z members
            this_om_position = this_om_geom.position
            for i, pulse in enumerate(entry[1]):
                if pulse.time in cleaned_time_list:
                    cleaned_mask = 1
                else:
                    cleaned_mask = 0
                # dom_x_temp = np.append(dom_x_temp, this_om_position.x)
                # dom_y_temp = np.append(dom_y_temp, this_om_position.y)
                # dom_z_temp = np.append(dom_z_temp, this_om_position.z)
                # dom_time_temp = np.append(dom_time_temp, pulse.time)
                # dom_charge_temp = np.append(dom_charge_temp, pulse.charge)
                # dom_lc_temp = np.append(dom_lc_temp, (pulse.flags & 0x1) >> 0)
                # dom_atwd_temp = np.append(
                #     dom_atwd_temp,
                #     (pulse.flags & 0x2) >> 1
                # )
                # dom_fadc_temp = np.append(
                #     dom_fadc_temp,
                #     (pulse.flags & 0x4) >> 2
                # )
                # dom_pulse_width_temp = np.append(
                #     dom_pulse_width_temp,
                #     pulse.width
                # )
                dom_key = str(this_om_key[0]) + '_' + str(this_om_key[1]) + '_' + str(this_om_key[2])
                pulses = np.array(
                    [
                        this_om_position.x,
                        this_om_position.y,
                        this_om_position.z,
                        pulse.time,
                        pulse.charge,
                        (pulse.flags & 0x1) >> 0,
                        (pulse.flags & 0x2) >> 1,
                        (pulse.flags & 0x4) >> 2,
                        pulse.width,
                        1,
                        cleaned_mask
                    ],
                    ndmin=2
                )
                temp = np.append(
                    temp,
                    pulses,
                    axis=0
                )
                dom_key_temp.append(dom_key)

        # Add Numpy arrays to data dictionary
        # data['dom_x'].append(dom_x_temp)
        # data['dom_y'].append(dom_y_temp)
        # data['dom_z'].append(dom_z_temp)
        # data['dom_time'].append(dom_time_temp)
        # data['dom_charge'].append(dom_charge_temp)
        # data['dom_lc'].append(dom_lc_temp)
        # data['dom_atwd'].append(dom_atwd_temp)
        # data['dom_fadc'].append(dom_fadc_temp)
        # data['dom_pulse_width'].append(dom_pulse_width_temp)
        # data['dom_key'].append(np.array(dom_key_temp))
        dom_key_temp = np.array(dom_key_temp)
        dom_key_temp = dom_key_temp[temp[:, 3].argsort()]
        data['dom_key'].append(dom_key_temp)
        temp = temp[temp[:, 3].argsort()]
        data['dom_x'].append(temp[:, 0])
        data['dom_y'].append(temp[:, 1])
        data['dom_z'].append(temp[:, 2])
        data['dom_time'].append(temp[:, 3])
        data['dom_charge'].append(temp[:, 4])
        data['dom_lc'].append(temp[:, 5])
        data['dom_atwd'].append(temp[:, 6])
        data['dom_fadc'].append(temp[:, 7])
        data['dom_pulse_width'].append(temp[:, 8])

        flatten = lambda l: [item for sublist in l for item in sublist]
        split_indices = np.argwhere(temp[:, 9])
        split_indices = np.array(flatten(split_indices))
        srt_indices = np.array(np.argwhere(temp[:, 10]))
        srt_indices = np.array(flatten(srt_indices))
        masks['SplitInIcePulses'].append(split_indices)
        masks['SRTInIcePulses'].append(srt_indices)

        # Get the cleaned pulses (if available)
        # <icecube.dataclasses.I3RecoPulseSeriesMap>
        # cleaned_pulses = frame['SRTInIcePulses'].apply(frame)

        # Get the LineFit reconstruction
        # The direction of the straight line
        # <icecube.dataclasses.I3Particle>
        linefit_direction = linefit_result.dir
        # Add the direction of the linefit to the data dictionary
        data['linefit_direction_x'].append(linefit_direction.x)
        data['linefit_direction_y'].append(linefit_direction.y)
        data['linefit_direction_z'].append(linefit_direction.z)

        # An arbitrary point along the line
        # <icecube.dataclasses.I3Particle>
        linefit_point_on_line = linefit_result.pos
        # Add the arbitrary point along the line to the data dictionary
        data['linefit_point_on_line_x'].append(linefit_point_on_line.x)
        data['linefit_point_on_line_y'].append(linefit_point_on_line.y)
        data['linefit_point_on_line_z'].append(linefit_point_on_line.z)

        # Some additional params
        # <icecube.recclasses.I3LineFitParams>
        # linefit_params = frame['LineFitParams']

        # Get the tensor of inertia
        # The direction of the ToI
        # <icecube.dataclasses.I3Particle>
        toi_direction = toi_result.dir
        # Add the direction to the data dictionary
        data['toi_direction_x'].append(toi_direction.x)
        data['toi_direction_y'].append(toi_direction.y)
        data['toi_direction_z'].append(toi_direction.z)

        # An arbitrary point along the line
        # <icecube.dataclasses.I3Particle>
        toi_point_on_line = toi_result.pos
        # Add arbitrary point along the line to data dictionary
        data['toi_point_on_line_x'].append(toi_point_on_line.x)
        data['toi_point_on_line_y'].append(toi_point_on_line.y)
        data['toi_point_on_line_z'].append(toi_point_on_line.z)

        # Some additional params
        # <icecube.recclasses.I3TensorOfInertiaFitParams>
        toi_params = frame['ToIParams']
        # This is the ratio of the smallest component of the ToI to the sum of
        # them all. A value close to 0. means a track-like event.
        # Addd it to the data dictionary
        data['toi_evalratio'].append(toi_params.evalratio)

        retro_crs_prefit_position = retro_crs_prefit_result.pos
        retro_crs_prefit_direction = retro_crs_prefit_result.dir

        data['retro_crs_prefit_x'].append(retro_crs_prefit_position.x)
        data['retro_crs_prefit_y'].append(retro_crs_prefit_position.y)
        data['retro_crs_prefit_z'].append(retro_crs_prefit_position.z)
        data['retro_crs_prefit_azimuth'].append(retro_crs_prefit_direction.azimuth)
        data['retro_crs_prefit_zenith'].append(retro_crs_prefit_direction.zenith)
        data['retro_crs_prefit_energy'].append(retro_crs_prefit_result.energy)
        data['retro_crs_prefit_time'].append(retro_crs_prefit_result.time)

    print('Converting i3 files:')
    db_train_file = save_dir.joinpath('train_set')
    with dumbdbm.open(str(db_train_file), 'n') as f:
        db_train = shelve.Shelf(f)
    db_val_file = save_dir.joinpath('val_set')
    with dumbdbm.open(str(db_val_file), 'n') as f:
        db_val = shelve.Shelf(f)
    db_test_file = save_dir.joinpath('test_set')
    with dumbdbm.open(str(db_test_file), 'n') as f:
        db_test = shelve.Shelf(f)
    n_events = 11308407
    indices_shuffled = np.arange(n_events)
    seed = 2912
    random.seed(seed)
    random.shuffle(indices_shuffled)
    train_indices_max = int(np.floor(0.8 * n_events))
    val_indices_max = int(n_events - (n_events - train_indices_max) * 0.5)
    train_dict = {}
    val_dict = {}
    test_dict = {}
    # db_file = save_dir.joinpath('oscnext-genie-level5-v01-01-pass2')
    # with dumbdbm.open(str(db_file), 'n') as f:
    #     db = shelve.Shelf(f)
    for i, file in enumerate(i3_files):
        # out_dict = {}
        out_name = file.stem
        out_name = out_name.replace('.i3', '')
        print('{}: converting {} --- {}/{}'.format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            file.stem,
            i + 1,
            len(i3_files))
        )
        data = {}
        masks = {}
        meta = {}

        for name in column_names:
            data['{0}'.format(name)] = []
        for name in mask_names:
            masks['{0}'.format(name)] = []
        for name in meta_names:
            meta['{0}'.format(name)] = []
        # Create the tray
        tray = I3Tray()

        # Read input file(s)
        tray.AddModule(
            'I3Reader',
            'reader',
            FilenameList=[str(gcd_file)] + [str(file)]
        )

        # Calculate tensor of interatia
        # TODO use clean or unclean pulses for InputReadout?
        tray.AddModule(
            'I3TensorOfInertia',
            'tensor_of_interia',
            AmplitudeOption=1,
            AmplitudeWeight=1,
            InputReadout='SRTInIcePulses',
            InputSelection='',
            MinHits=3,
            Name='ToI'
        )

        tray.AddModule(
            hit_statistics.I3HitStatisticsCalculator,
            PulseSeriesMapName='SRTInIcePulses',
            OutputI3HitStatisticsValuesName='HitStatisticsValues',
        )
        tray.AddModule(
            hit_multiplicity.I3HitMultiplicityCalculator,
            PulseSeriesMapName='SRTInIcePulses',
            OutputI3HitMultiplicityValuesName='HitMultiplicityValues'
        )
        tray.AddModule(
            time_characteristics.I3TimeCharacteristicsCalculator,
            PulseSeriesMapName='SRTInIcePulses',
            OutputI3TimeCharacteristicsValuesName='TimeCharacteristicsValues'
        )

        # Add our own module
        tray.Add(do_things, 'do_things')
        # Actually run the tray
        tray.Execute()
        tray.Finish()

        events_list = []
        masks_list = []
        meta_list = []

        no_of_events = len(data['dom_x'])
        for j in range(no_of_events):
            temp_events = {key: data[key][j] for key in data}
            temp_masks = {key: masks[key][j] for key in masks}
            temp_meta = {
                'file': out_name + '.h5',
                'index': j,
                'particle_code': file.stem.split('.')[2],
                'level': 5
            }
            events_list.append(temp_events)
            masks_list.append(temp_masks)
            meta_list.append(temp_meta)

        taken_indices = indices_shuffled[0:no_of_events]
        indices_shuffled = np.delete(indices_shuffled, np.s_[0:no_of_events])
 
        for j, i_index in enumerate(taken_indices):
            if i_index <= train_indices_max:
                train_dict[str(i_index)] = {} 
                train_dict[str(i_index)]['raw'] = events_list[j]
                train_dict[str(i_index)]['masks'] = masks_list[j]
                train_dict[str(i_index)]['meta'] = meta_list[j]
            elif i_index >= train_indices_max and i_index <= val_indices_max:
                val_dict[str(i_index)] = {} 
                val_dict[str(i_index)]['raw'] = events_list[j]
                val_dict[str(i_index)]['masks'] = masks_list[j]
                val_dict[str(i_index)]['meta'] = meta_list[j]
            else:
                test_dict[str(i_index)] = {} 
                test_dict[str(i_index)]['raw'] = events_list[j]
                test_dict[str(i_index)]['masks'] = masks_list[j]
                test_dict[str(i_index)]['meta'] = meta_list[j]

        # out_dict = data
        # out_dict['masks'] = masks
        # out_dict['meta'] = {}
        # out_dict['meta']['events'] = len(data['true_primary_energy'])

        # db[out_name] = out_dict

        if (i + 1) % 20 == 0:
            print('{}: saving to db'.format(
                datetime.datetime.now().strftime("%H:%M:%S")
            ))

            with shelve.open(str(db_train_file), 'w') as f:
                for key in train_dict:
                    f[key] = train_dict[key]
            with shelve.open(str(db_val_file), 'w') as f:
                for key in val_dict:
                    f[key] = val_dict[key]
            with shelve.open(str(db_test_file), 'w') as f:
                for key in test_dict:
                    f[key] = test_dict[key]
            train_dict = {}
            val_dict = {}
            test_dict = {}

    print('{}: saving to db'.format(
        datetime.datetime.now().strftime("%H:%M:%S")
    ))

    with shelve.open(str(db_train_file), 'w') as f:
        for key in train_dict:
            f[key] = train_dict[key]
    with shelve.open(str(db_val_file), 'w') as f:
        for key in val_dict:
            f[key] = val_dict[key]
    with shelve.open(str(db_test_file), 'w') as f:
        for key in test_dict:
            f[key] = test_dict[key]
    train_dict = {}
    val_dict = {}
    test_dict = {}

print('Done.')