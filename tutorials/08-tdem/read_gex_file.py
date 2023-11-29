import numpy as np
import io

def read_gex_file(fname, n_gates=37):
    fid = io.open(fname, mode="r", encoding="utf-8")
    lines = fid.readlines()
    fid.close()
    lm_waveform = []
    hm_waveform = []
    time_gates = []
    for i_count, line in enumerate(lines):
        if 'TxLoopArea' in line:
            tx_area = float(line.split('=')[-1])
        if 'WaveformLMPoint' in line:
            tmp = line.split()
            lm_waveform.append(np.array([tmp[-2], tmp[-1]], dtype=float))
        if 'WaveformHMPoint' in line:
            tmp = line.split()
            hm_waveform.append(np.array([tmp[-2], tmp[-1]], dtype=float))
        if 'GateTime' in line:
            time_gates.append(line.split('=')[1].split()[-3:])
        if '[Channel1]' in line:
            metadata_ch1 = lines[i_count+1:i_count+17]
        if '[Channel2]' in line:
            metadata_ch2 = lines[i_count+1:i_count+17]
    lm_waveform = np.vstack(lm_waveform)
    hm_waveform = np.vstack(hm_waveform)
    lm_waveform = lm_waveform[np.argwhere(lm_waveform[:,1]==0.)[2][0]:,:]
    hm_waveform = hm_waveform[np.argwhere(hm_waveform[:,1]==0.)[2][0]:,:]
    time_gates = np.vstack(time_gates[:n_gates]).astype(float)
    dict_ch1 = {}
    for dat in metadata_ch1:
        if dat.split() != []:
            tmp = dat.split()[0].split('=')
        try:
            dict_ch1[tmp[0]] = float(tmp[1])
        except:
            dict_ch1[tmp[0]] = tmp[1]
    dict_ch1['waveform'] = lm_waveform
    dict_ch2 = {}
    for dat in metadata_ch2:
        if dat.split() != []:
            tmp = dat.split()[0].split('=')
            try:
                dict_ch2[tmp[0]] = float(tmp[1])
            except:
                dict_ch2[tmp[0]] = tmp[1]
    dict_ch2['waveform'] = hm_waveform
    df_time = pd.DataFrame(data=time_gates, columns=['center', 'start', 'end'])
    output_dict = {'lm': dict_ch1, 'hm': dict_ch2, 'time_gates':df_time, 'tx_area':tx_area}
    return output_dict
