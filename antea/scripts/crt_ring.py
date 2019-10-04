import sys
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.utils.table_functions import load_rpos
from antea.io   .mc_io           import load_mchits
from antea.io   .mc_io           import load_mcparticles
from antea.io   .mc_io           import load_mcsns_response
from antea.io   .mc_io           import load_mcTOFsns_response

### read sensor positions from database
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

start   = int  (sys.argv[1])
numb    = int  (sys.argv[2])
thr_r   = float(sys.argv[3])
thr_phi = float(sys.argv[4])
thr_z   = float(sys.argv[5])
thr_e   = float(sys.argv[6])

folder    = '/folder_path/'
file_full = folder + 'input_file_name.{0:03d}.pet.h5'
evt_file  = '/folder_path/output_file_name.{0}_{1}_{2}_{3}_{4}_{5}'.format(start, numb, int(thr_r), int(thr_phi), int(thr_z), int(thr_e))

rpos_file = '/map_folder_path/r_table_name.h5'

Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = "f4pes200bins")

time_diff = []
pos_cart1 = []
pos_cart2 = []
event_ids = []

ave_speed_in_LXe = 0.210 # mm/ps

for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response     = load_mcsns_response(file_name)
        sns_response_tof = load_mcTOFsns_response(file_name)
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/waveforms in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    particles = load_mcparticles(file_name)
    hits      = load_mchits(file_name)
    
    events = particles.event_id.unique()

    for evt in events[:]:

        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if len(true_pos) < 2: continue ## Only coincidences

        sns_evt     = sns_response    [sns_response    .event_id == evt]
        sns_evt_tof = sns_response_tof[sns_response_tof.event_id == evt]

        sns_resp_r   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_r)
        sns_resp_phi = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_phi)
        sns_resp_z   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_z)
        sns_resp_e   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_e)

        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_r, true_pos, DataSiPM_idx)
        r1 = r2 = None
        if len(pos1) > 0 and len(pos2) > 0:
            pos1_phi  = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            diff_sign1 = min(pos1_phi) < 0 < max(pos1_phi)
            if diff_sign1 & (np.abs(np.min(pos1_phi))>np.pi/2.):
                pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
            mean_phi1 = np.average(pos1_phi, weights=q1)
            var_phi1  = np.average((pos1_phi-mean_phi1)**2, weights=q1)
            r1        = Rpos(np.sqrt(var_phi1)).value

            pos2_phi  = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            diff_sign2 = min(pos2_phi ) < 0 < max(pos2_phi)
            if diff_sign2 & (np.abs(np.min(pos2_phi))>np.pi/2.):
                pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
            mean_phi2 = np.average(pos2_phi, weights=q2)
            var_phi2  = np.average((pos2_phi-mean_phi2)**2, weights=q2)
            r2        = Rpos(np.sqrt(var_phi2)).value


        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_phi, true_pos, DataSiPM_idx)
        phi1 = phi2 = None
        if len(pos1) > 0 and len(pos2) > 0:
            reco_cart1 = np.average(pos1, weights=q1, axis=0)
            reco_cart2 = np.average(pos2, weights=q2, axis=0)
            phi1       = np.arctan2(reco_cart1[1], reco_cart1[0])
            phi2       = np.arctan2(reco_cart2[1], reco_cart2[0])


        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_z, true_pos, DataSiPM_idx)
        z1 = z2 = None
        if len(pos1) > 0 and len(pos2) > 0:
            reco_cart1 = np.average(pos1, weights=q1, axis=0)
            reco_cart2 = np.average(pos2, weights=q2, axis=0)
            z1         = reco_cart1[2]
            z2         = reco_cart2[2]

        q1, q2, _, _ = rf.assign_sipms_to_gammas(sns_resp_e, true_pos, DataSiPM_idx)


        pos1_cart = []
        pos2_cart = []
        if r1 and phi1 and z1 and q1 and r2 and phi2 and z2 and q2:
            pos1_cart.append(r1 * np.cos(phi1))
            pos1_cart.append(r1 * np.sin(phi1))
            pos1_cart.append(z1)
            pos2_cart.append(r2 * np.cos(phi2))
            pos2_cart.append(r2 * np.sin(phi2))
            pos2_cart.append(z2)
        a_cart1 = np.array(pos1_cart)
        a_cart2 = np.array(pos2_cart)


        first_timestamp_tof = sns_evt_tof.groupby(['event_id','sensor_id'])[['time_bin']].min()
        first_timestamp_tof = first_timestamp_tof.reset_index()
        t1, t2, pos1_tof, pos2_tof = rf.assign_sipms_to_gammas_tof(first_timestamp_tof, true_pos, DataSiPM_idx)

        min_time_1 = min(t1)
        min_time_2 = min(t2)
        min_pos_1  = pos1_tof[t1.index(min_time_1)]
        min_pos_2  = pos2_tof[t2.index(min_time_2)]
        min_time   = min_time_1 - min_time_2


        ### Distance between interaction point and sensor detecting first photon
        dist1 = np.linalg.norm(a_cart1 - min_pos_1)
        dist2 = np.linalg.norm(a_cart2 - min_pos_2)
        dist  = dist1 - dist2

        delta_t = 1/2 *(dist/ave_speed_in_LXe - min_time)

        time_diff.append(delta_t)
        pos_cart1.append(a_cart1)
        pos_cart2.append(a_cart2)
        event_ids.append(evt)


a_time_diff = np.array(time_diff)
a_pos_cart1 = np.array(pos_cart1)
a_pos_cart2 = np.array(pos_cart2)
a_event_ids = np.array(event_ids)

np.savez(evt_file, a_time_diff=a_time_diff, a_pos_cart1=a_pos_cart1, a_pos_cart2=a_pos_cart2, a_event_ids=a_event_ids)


