import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import wandb
import torch
import pickle
#from data.visua.autoenco import DeepAutoencoder_connect

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        #self.AE = DeepAutoencoder_connect().cuda(2)
        #self.AE.load_state_dict(torch.load('data/visua/weights/best_DeepAutoencoder_connect0710.pth'))
        #self.AE.eval()
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['data']
            self.label = npz_data['labels']
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['data']
            self.label = npz_data['labels']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        
        
        # self.data = np.reshape(self.data, (1,300,150))
        
        N, T, _, _,_ = self.data.shape
       # print(self.data.shape) # (18932, 3, 64, 25, 2)
       # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        # data_numpy = np.expand_dims(data_numpy, axis=0)
        # data_numpy= torch.Tensor(data_numpy).cuda(2)
        # data_numpy = self.AE(data_numpy)
        
        # data_numpy = data_numpy.cpu().detach().numpy()
        # data_numpy = np.squeeze(data_numpy)

        data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot: #False
            data_numpy = tools.random_rot(data_numpy)
        if self.bone: #False
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel: #False
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        self.label[index] = label
        
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def process_and_save(self, output_path = 'bonelength_test_data.npz'):
        all_data_numpy = np.array(self.data)
        all_labels = np.array(self.label)

        valid_frame_nums = [np.sum(sample.sum(0).sum(-1).sum(-1) != 0) for sample in all_data_numpy]

        processed_data = tools.valid_crop_resize_for_data_save_batch(all_data_numpy, valid_frame_nums, self.p_interval, self.window_size)

        np.savez(output_path, data=processed_data, labels=all_labels)
        print(f"Data saved to {output_path}")


class Feeder_attack(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        #self.AE = DeepAutoencoder_connect().cuda(2)
        #self.AE.load_state_dict(torch.load('data/visua/weights/best_DeepAutoencoder_connect0710.pth'))
        #self.AE.eval()
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = np.load('./adv_data/ntu60/cw_train.npy',allow_pickle=True)
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            #self.data = npz_data['x_test']
            #self.label = np.where(npz_data['y_test'] > 0)[1]
            self.data = npz_data
            self.label = np.load('data/ntu_attack/iter_100_thres0.30label.pkl', allow_pickle=True)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        
        
        # self.data = np.reshape(self.data, (1,300,150))

        
        if self.split == 'train':
            N, C, T, V, M = self.data.shape
            self.data = self.data.reshape((N, T, M, V, C)).transpose(0, 4, 1, 3, 2)
        
        if self.split == 'test':
            N, T, _,_,_ = self.data.shape
            #self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        #N, T, _ = self.data.shape
        #self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        # data_numpy = np.expand_dims(data_numpy, axis=0)
        # data_numpy= torch.Tensor(data_numpy).cuda(2)
        # data_numpy = self.AE(data_numpy)
        
        # data_numpy = data_numpy.cpu().detach().numpy()
        # data_numpy = np.squeeze(data_numpy)

        data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot: #False
            data_numpy = tools.random_rot(data_numpy)
        if self.bone: #False
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel: #False
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        self.label[index] = label
        
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def process_and_save(self, output_path = 'bonelength_test_data.npz'):
        all_data_numpy = np.array(self.data)
        all_labels = np.array(self.label)

        valid_frame_nums = [np.sum(sample.sum(0).sum(-1).sum(-1) != 0) for sample in all_data_numpy]

        processed_data = tools.valid_crop_resize_for_data_save_batch(all_data_numpy, valid_frame_nums, self.p_interval, self.window_size)

        np.savez(output_path, data=processed_data, labels=all_labels)
        print(f"Data saved to {output_path}")
    
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
