import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import torch

class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx

def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, mode, num_pc,
                 P_tx_amplitude=5, P_ty_amplitude=5, P_tz_amplitude=5,
                 P_Rx_amplitude=0, P_Ry_amplitude=2.0 * math.pi, P_Rz_amplitude=0,num_kpt=512,is_front=False):
        super(kitti_pc_img_dataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.dataset = self.make_kitti_dataset(root_path, mode)
        self.calibhelper = KittiCalibHelper(root_path)
        self.num_pc = num_pc
        self.img_H = 160
        self.img_W = 512

        self.P_tx_amplitude = P_tx_amplitude
        self.P_ty_amplitude = P_ty_amplitude
        self.P_tz_amplitude = P_tz_amplitude
        self.P_Rx_amplitude = P_Rx_amplitude
        self.P_Ry_amplitude = P_Ry_amplitude
        self.P_Rz_amplitude = P_Rz_amplitude
        self.num_kpt=num_kpt
        self.farthest_sampler = FarthestSampler(dim=3)

        self.node_a_num= 128
        self.node_b_num= 128
        self.is_front=is_front
        print('load data complete')

    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        elif 'tiny_val' == mode:
            seq_list = [9]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_2')
            img3_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_3')
            pc_folder = os.path.join(root_path, 'data_odometry_velodyne', 'sequences', '%02d' % seq, 'voxel0.1-SNr0.6')


            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                seq, i, 'P3', sample_num))
        return dataset


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)
        pcd.normals=o3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        counter = 0
        while len(pc_np_list) < 3:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0
            sn_np = transform_pc_np(P_transform_rot, sn_np)

            pc_np_list.append(pc_np)
            intensity_np_list.append(intensity_np)
            sn_np_list.append(sn_np)

        return pc_np_list, intensity_np_list, sn_np_list

    def get_pointcloud(self, pc_folder, seq_i):
        pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        sn_np = npy_data[4:7, :]  # 3xN

        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R


    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_folder, pc_folder, seq, seq_i, key, seq_sample_num = self.dataset[index]

        # ----------------------load image and points----------------
        img_path = os.path.join(img_folder, '%06d.png' % seq_i)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # HxWx3

        # load point cloud of seq_i
        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))
        pc, intensity, sn = self.get_pointcloud(pc_folder, seq_i)
        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]
        sn = np.dot(P_Tr[0:3, 0:3], sn)


        # -------------------process points------------------------
        pc, intensity, sn = self.downsample_with_intensity_sn(pc, intensity, sn, voxel_grid_downsample_size=0.1)

        pc, intensity, sn = self.downsample_np(pc, intensity,sn)

        # -------------------process pixels------------------------
        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]

        if 'train' == self.mode:
            img = self.augment_img(img)

        # -----------------get in_matrix --------------------------
        K = self.calibhelper.get_matrix(seq, key + '_K')
        K = self.camera_matrix_scaling(K, 0.5)
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #1/4 scale
        K_4=self.camera_matrix_scaling(K,0.25)

        # -------------------------------------------------------

        
        pc_ = np.dot(K_4, pc)
        pc_mask = np.zeros((1, np.shape(pc)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.floor(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) & (xy[1, :] >= 0) & (xy[1, :] <= (self.img_H*0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.

        pc_kpt_idx=np.where(pc_mask.squeeze()==1)[0]
        index=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        pc_kpt_idx=pc_kpt_idx[index]

        pc_outline_idx=np.where(pc_mask.squeeze()==0)[0]
        index=np.random.permutation(len(pc_outline_idx))[0:self.num_kpt]
        pc_outline_idx=pc_outline_idx[index]

        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*0.25), int(self.img_W*0.25))).toarray()
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1.

        img_kpt_index=xy[1,pc_kpt_idx]*self.img_W*0.25 +xy[0,pc_kpt_idx]


        img_outline_index=np.where(img_mask.squeeze().reshape(-1)==0)[0]
        index=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
        img_outline_index=img_outline_index[index]

        P = self.generate_random_transform()

        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        sn = np.dot(P[0:3, 0:3], sn)

        node_a_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_a_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_a_num)

        node_b_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_b_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_b_num)

        if pc_kpt_idx.shape[0] < 512:
            print(pc_kpt_idx.shape)
        if img_kpt_index.shape[0] < 512:
            print(img_kpt_index.shape)

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'intensity': torch.from_numpy(intensity.astype(np.float32)),
                'sn': torch.from_numpy(sn.astype(np.float32)),
                'K': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),

                'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,40960)
                'img_mask': torch.from_numpy(img_mask).float(),     #(40,128)
                
                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #512
                'pc_outline_idx':torch.from_numpy(pc_outline_idx),  #512
                'img_kpt_idx':torch.from_numpy(img_kpt_index).long() ,      #512
                'img_outline_index':torch.from_numpy(img_outline_index).long(),
                'node_a':torch.from_numpy(node_a_np).float(),
                'node_b':torch.from_numpy(node_b_np).float()
                }
               

def projection_pc_img(pc_np, img, K, size=1):
    """

    :param pc_np: points in camera coordinate
    :param img: image of the same frame
    :param K: Intrinsic matrix
    :return:
    """
    img_vis = np.copy(img)
    H, W = img.shape[0], img.shape[1]

    pc_np_front = pc_np[:, pc_np[2, :]>1.0]  # 3xN

    pc_pixels = np.dot(K, pc_np_front)  # 3xN
    pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
    for i in range(pc_pixels.shape[1]):
        px = int(pc_pixels[0, i])
        py = int(pc_pixels[1, i])
        # determine a point on image plane
        if px>=0 and px<=W-1 and py>=0 and py<=H-1:
            cv2.circle(img_vis, (px, py), size, (255, 0, 0), -1)
    return img_vis

if __name__ == '__main__':
    dataset = kitti_pc_img_dataset('/PATH', 'val', 40960)
    data = dataset[4000]
    
    print(len(dataset))
    print(data['K'])
    print(data['pc'].size())
    print(data['img'].size())
    print(data['pc_mask'].size())
    print(data['intensity'].size())
    print(data['node_a'].size())
    print(data['node_b'].size())
    print(data['pc_kpt_idx'].size())
