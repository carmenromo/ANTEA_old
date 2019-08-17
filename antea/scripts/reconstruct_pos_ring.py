import sys
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.utils.table_functions import load_rpos

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

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

photo_response1, photo_response2 = [], []
touched_sipms1,  touched_sipms2  = [], []
event_ids = []

for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response = pd.read_hdf(file_name, 'MC/waveforms')
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

    particles = pd.read_hdf(file_name, 'MC/particles')
    hits      = pd.read_hdf(file_name, 'MC/hits')
    
    events = particles.event_id.unique()

    for evt in events[:]:

        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if not select: continue

        sns_evt = sns_response[sns_response.event_id == evt]

        sns_resp_r   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_r)
        sns_resp_phi = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_phi)
        sns_resp_z   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_z)
        sns_resp_e   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_e)
               
        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_r, true_pos, DataSiPM_idx)
        r1 = r2 = None
        if len(pos1) > 0:
            pos1_phi  = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            diff_sign = min(pos1_phi ) < 0 < max(pos1_phi)
            if diff_sign & (np.abs(np.min(pos1_phi))>np.pi/2.):
                pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
            mean_phi = np.average(pos1_phi, weights=q1)
            var_phi1 = np.average((pos1_phi-mean_phi)**2, weights=q1)
            r1       = Rpos(np.sqrt(var_phi1)).value
        if len(pos2) > 0:
            pos2_phi  = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            diff_sign = min(pos2_phi ) < 0 < max(pos2_phi)
            if diff_sign & (np.abs(np.min(pos2_phi))>np.pi/2.):
                pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
            mean_phi = np.average(pos2_phi, weights=q2)
            var_phi2 = np.average((pos2_phi-mean_phi)**2, weights=q2)
            r2       = Rpos(np.sqrt(var_phi2)).value


        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_phi, true_pos, DataSiPM_idx)
        phi1 = phi2 = None
        if len(pos1) > 0:
            reco_cart_pos = np.average(pos1, weights=q1, axis=0)
            phi1          = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
        if len(pos2) > 0:
            reco_cart_pos = np.average(pos2, weights=q2, axis=0)
            phi2          = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])


        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(sns_resp_z, true_pos, DataSiPM_idx)
        z1 = z2 = None
        if len(pos1) > 0:
            reco_cart_pos = np.average(pos1, weights=q1, axis=0)
            z1            = reco_cart_pos[2]
        if len(pos2) > 0:
            reco_cart_pos = np.average(pos2, weights=q2, axis=0)
            z2            = reco_cart_pos[2]

        q1, q2, _, _ = rf.assign_sipms_to_gammas(sns_resp_e, true_pos, DataSiPM_idx)


        if r1 and phi1 and z1 and q1:
            reco_r1        .append(r1)
            reco_phi1      .append(phi1)
            reco_z1        .append(z1)
            true_r1        .append(np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2))
            true_phi1      .append(np.arctan2(true_pos[0][1], true_pos[0][0]))
            true_z1        .append(true_pos[0][2])
            photo_response1.append(sum(q1))
            touched_sipms1 .append(len(q1))
            event_ids      .append(evt)
        else:
            reco_r1        .append(1.e9)
            reco_phi1      .append(1.e9)
            reco_z1        .append(1.e9)
            true_r1        .append(1.e9)
            true_phi1      .append(1.e9)
            true_z1        .append(1.e9)
            photo_response1.append(1.e9)
            touched_sipms1 .append(1.e9)
            event_ids      .append(evt)


        if r2 and phi2 and z2 and q2:
            reco_r2        .append(r2)
            reco_phi2      .append(phi2)
            reco_z2        .append(z2)
            true_r2        .append(np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2))
            true_phi2      .append(np.arctan2(true_pos[1][1], true_pos[1][0]))
            true_z2        .append(true_pos[1][2])
            photo_response2.append(sum(q1))
            touched_sipms2 .append(len(q1))
        else:
            reco_r2        .append(1.e9)
            reco_phi2      .append(1.e9)
            reco_z2        .append(1.e9)
            true_r2        .append(1.e9)
            true_phi2      .append(1.e9)
            true_z2        .append(1.e9)
            photo_response2.append(1.e9)
            touched_sipms2 .append(1.e9)


a_true_r1         = np.array(true_r1)
a_true_phi1       = np.array(true_phi1)
a_true_z1         = np.array(true_z1)
a_reco_r1         = np.array(reco_r1)
a_reco_phi1       = np.array(reco_phi1)
a_reco_z1         = np.array(reco_z1)
a_photo_response1 = np.array(photo_response1)
a_touched_sipms1  = np.array(touched_sipms1)

a_true_r2         = np.array(true_r2)
a_true_phi2       = np.array(true_phi2)
a_true_z2         = np.array(true_z2)
a_reco_r2         = np.array(reco_r2)
a_reco_phi2       = np.array(reco_phi2)
a_reco_z2         = np.array(reco_z2)
a_photo_response2 = np.array(photo_response2)
a_touched_sipms2  = np.array(touched_sipms2)

a_event_ids       = np.array(event_ids)


np.savez(evt_file, a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1, a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2, a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1, a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,  a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, a_photo_response1=a_photo_response1, a_photo_response2=a_photo_response2, a_event_ids=a_event_ids)
            


        



