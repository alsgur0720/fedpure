import numpy as np
from einops import rearrange
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation


def normalize_time_series(time_series, start, end):
    if end - start == 0:
        normalized_time = np.array([0])
        return normalized_time, time_series[start:end+1]
    else:
        normalized_time = np.linspace(0, 1, end - start + 1)
        return normalized_time, time_series[start:end+1]

def find_action_bounds_ori(ts1, ts2, threshold=0.5):
    start_1, end_1 = 0, len(ts1) - 1
    start_2, end_2 = 0, len(ts2) - 1

    for i in range(1, len(ts1)):
        # print(frame_similarity(ts1[i], ts1[0]))
        if frame_similarity(ts1[i], ts1[0]) > threshold:
            start_1 = i
            break


    for i in range(1, len(ts2)):
        # print(frame_similarity(ts2[i], ts2[0]))
        if frame_similarity(ts2[i], ts2[0]) > threshold:
            start_2 = i
            break


    for i in range(len(ts1) - 2, -1, -1):
        if frame_similarity(ts1[i], ts1[-1]) > threshold:
            end_1 = i
            break


    for i in range(len(ts2) - 2, -1, -1):
        if frame_similarity(ts2[i], ts2[-1]) > threshold:
            end_2 = i
            break

    return start_1, end_1, start_2, end_2

def frame_similarity(frame1, frame2):
    return norm(frame1 - frame2)

def find_action_bounds_simple(time_series):
    start = 0
    end = len(time_series) - 1
    return start, end

def find_action_bounds(ts1, ts2, threshold=0.5):
    
    start_1, end_1 = 0, len(ts1) - 1
    start_2, end_2 = 0, len(ts2) - 1
    
    start_1 = np.argmax([frame_similarity(abs(ts1[i]), abs(ts1[0])) for i in range(1, len(ts1))])

    start_2 = np.argmax([frame_similarity(abs(ts2[i]), abs(ts2[0])) for i in range(1, len(ts2))])
    

    for i in range(start_1, len(ts1) - 2):
        if frame_similarity(ts1[i], ts1[i+1]) < threshold:
            end_1 = i
            break

    for i in range(start_2, len(ts2) - 1):
        if frame_similarity(ts2[i], ts2[i+1]) < threshold:
            end_2 = i
            break
    
    # if end_1 - start_1 < 0:
    #     end_1 = len(ts1) - 1
        
    # if end_2 - start_2 < 0:
    #     end_2 = len(ts2) - 1
        
        
    return start_1, end_1, start_2, end_2

def calculate_angle(skeleton, joint_indices):

    ba = skeleton[:, joint_indices[1], :] - skeleton[:, joint_indices[0], :]
    bc = skeleton[:, joint_indices[2], :] - skeleton[:, joint_indices[1], :]
    

    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    

    valid_indices = (norm_ba > 1e-8) & (norm_bc > 1e-8)  
    

    cosine_angle = np.zeros_like(norm_ba)  
    cosine_angle[valid_indices] = np.einsum('ij,ij->i', ba[valid_indices], bc[valid_indices]) / (norm_ba[valid_indices] * norm_bc[valid_indices])
    

    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  
    angle = np.degrees(angle)  
    
    return angle


# for i in range(100):
#     print(label[i], "{}".format(i))
# exit()

# count = 0

# for i in range(len(my_data_clean)):
# # for i in range(10):
#     if label[i] != 6:
#         pass
#     else:
#         count +=1

def Data_poisoning_attack_detector(transform_prototype, velocity, reference_frame):
    save = []
    max_corrs = {}
    max_corr_indexes = {}
    for i in range(len(transform_prototype)):
        for j in range(len(transform_prototype)):
            
            
            
            data = np.zeros((300, 25, 3))
            data[0,:,:] = reference_frame @ transform_prototype
            data[1:, :, :] = velocity

            for j in range(0, 299):
                if (np.all(velocity[j,:,:] == 0)) == True:
                    data[j,:,:] = 0
                else:
                    data[j+1,:,:] = data[j,:,:] + data[j+1,:,:]

            
            data0 = rearrange(data[i][:,:,:,0], 'x y z -> y z x')
            data1 = rearrange(data[j][:,:,:,0], 'x y z -> y z x')



            
            
            non_zero_frames0 = np.any(data0.reshape(300, -1), axis=1)
            non_zero_frames1 = np.any(data1.reshape(300, -1), axis=1)
            time_series_1 = data0[non_zero_frames0]
            time_series_2 = data1[non_zero_frames1]


            start_1, end_1, start_2, end_2 = find_action_bounds_ori(time_series_1, time_series_2)

            normalized_time_1, normalized_series_1 = normalize_time_series(time_series_1, start_1, end_1)
            normalized_time_2, normalized_series_2 = normalize_time_series(time_series_2, start_2, end_2)


            common_length = min(len(normalized_series_1), len(normalized_series_2))
            common_time = np.linspace(0, 1, common_length)


            similarities = []
            saved_frames_1 = []
            saved_frames_2 = []
            
            
            for t1 in range(common_length):
                first_frame1 = normalized_series_1[t1]
                similarities = []
                for t2 in range(common_length):
                    frame_t = normalized_series_2[t2]
                    # 프레임 간의 L2 거리 계산
                    distance = np.linalg.norm(first_frame1 - frame_t)
                    similarities.append(1 / distance)
                max_similarity_index = np.argmax(similarities)
                saved_frames_1.append(first_frame1)
                saved_frames_2.append(normalized_series_2[max_similarity_index])
            
            
                
            normalized_series_1 = np.array(saved_frames_1)
            normalized_series_2 = np.array(saved_frames_2)
            
                
            interp_series_1_ori = np.sum(normalized_series_1,axis=1)
            # interp_series_1_inter = np.sum(interp_series_1 ,axis=1)
            
            
            interp_series_2_ori = np.sum(normalized_series_2,axis=1)
            # interp_series_2_inter = np.sum(interp_series_2,axis=1)
            
            increasing_array_1 = np.arange(0, len(interp_series_1_ori))
            increasing_array_2 = np.arange(0, len(interp_series_2_ori))
            # increasing_array = np.arange(0, len(interp_series_2_inter))
            
            correlations = []
            # for t in range (normalized_series_1.shape[0]):
            for dim_index in range(3):
                correlations_per_dim = []
                for joint_index in range(25):
                    series1 = normalized_series_1[:, joint_index, dim_index]
                    series2 = normalized_series_2[:, joint_index, dim_index]
                    corr = np.corrcoef(series1, series2)[0, 1]
                    corr = np.abs(corr)
                    correlations_per_dim.append(corr)
                correlations.append(np.mean(correlations_per_dim))
                

            
            max_corr = max(correlations)

            max_corrs[i] = max_corr
            max_corr_index = correlations.index(max_corr)
            max_corr_indexes[i] = max_corr_index
            
            if max_corr_index == 0:
                max_corr_coord = 'x'
            elif max_corr_index == 1:
                max_corr_coord = 'y'
            else:
                max_corr_coord = 'z'



    keys_below_threshold = [key for key, value in max_corrs.items() if value == None or value <= 0.5]

    mode_value = max(set(max_corr_indexes.values()), key=list(max_corr_indexes.values()).count)


    keys_not_mode = [key for key, value in max_corr_indexes.items() if value != mode_value]
    # print("Keys in max_corr_indexes dictionary with values different from the mode:", keys_not_mode)
            

    # Find common values between two lists
    common_values = set(keys_below_threshold) & set(keys_not_mode)
    
    return common_values

