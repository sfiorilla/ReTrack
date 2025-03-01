import numpy as np
import os
import glob

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)


def dti(txt_path, save_path, n_min=25, n_dti=20):
    assert os.path.exists(txt_path), "File path Not Found"
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    print(f"Found {len(seq_txts)} sequences:",flush=True)
    for seq_txt in seq_txts:
        print(seq_txt) ### extra added
        seq_name = seq_txt.split('/')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)


if __name__ == '__main__':
    #data_root = '/home/tstanczy/repos/ByteTrack/datasets/mot/train'
    #data_root = '/home/tstanczy/repos/ByteTrack/datasets/MOT20/train'
    txt_path = '/home/sfiorill/ReTrack/TrackEval/data/trackers/mot_challenge/MOT17-test/last_execution/data/'
    save_path = '/home/sfiorill/ReTrack/TrackEval/data/trackers/mot_challenge/MOT17-test/last_execution/data_dti/'
    

    mkdir_if_missing(save_path)
    dti(txt_path, save_path, n_min=5, n_dti=20)

    # print('Before DTI: ')
    # eval_mota(data_root, txt_path)
    # print('After DTI:')
    # eval_mota(data_root, save_path)

    '''
    mota_best = 0.0
    best_n_min = 0
    best_n_dti = 0
    for n_min in range(5, 50, 5):
        for n_dti in range(5, 30, 5):
            dti(txt_path, save_path, n_min, n_dti)
            mota = get_mota(data_root, save_path)
            if mota > mota_best:
                mota_best = mota
                best_n_min = n_min
                best_n_dti = n_dti
                print(mota_best, best_n_min, best_n_dti)
    print(mota_best, best_n_min, best_n_dti)
    '''
