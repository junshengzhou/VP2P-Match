import os
import torch
import argparse
from models.triplet_network import VP2PMatchNet
from models.kitti_dataset import kitti_pc_img_dataset
import utils.loss as loss
import numpy as np
import datetime
import logging
import math
import utils.options as options
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import random

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

def test_acc_trans(model,testdataloader,args):

    test_name = args.exp_name
    t_diff_set=[]
    angles_diff_set=[]

    for step,data in enumerate(testdataloader):

        if step%1==0:
            model.eval()
            img=data['img'].cuda()              #full size
            pc_all=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K_all=data['K'].cuda()
            P_all=data['P'].cuda()

            img_features_all,pc_features_all,img_score_all,pc_score_all=model(pc_all,intensity,sn,img)     #64 channels feature
            
            bs = img_score_all.shape[0]

            for i in range(bs):
                img_score=img_score_all[i]
                pc_score=pc_score_all[i]
                img_feature=img_features_all[i]
                pc_feature=pc_features_all[i]
                pc=pc_all[i]
                P=P_all[i].data.cpu().numpy()
                K=K_all[i].data.cpu().numpy()
                
                img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
                img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

                img_xy=np.concatenate((img_x,img_y),axis=0)
                img_xy = torch.tensor(img_xy).cuda()

                img_xy_flatten=img_xy.reshape(2,-1)
                img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
                img_score_flatten=img_score.squeeze().reshape(-1)

                img_index=(img_score_flatten>args.img_thres)

                img_xy_flatten_sel=img_xy_flatten[:,img_index]
                img_feature_flatten_sel=img_feature_flatten[:,img_index]
                img_score_flatten_sel=img_score_flatten[img_index]
                if img_xy_flatten_sel.shape[1] > 2800:
                    img_xy_flatten_sel = img_xy_flatten_sel[:,:2800]
                    img_feature_flatten_sel = img_feature_flatten_sel[:,:2800]

                pc_index=(pc_score.squeeze()>args.pc_thres)

                pc_sel=pc[:,pc_index]
                pc_feature_sel=pc_feature[:,pc_index]
                pc_score_sel=pc_score.squeeze()[pc_index]
                if pc_sel.shape[1] > 8500:
                    pc_sel = pc_sel[:,:8500]
                    pc_feature_sel = pc_feature_sel[:,:8500]

                dist=1-torch.sum(img_feature_flatten_sel.unsqueeze(2)*pc_feature_sel.unsqueeze(1), dim=0)
                sel_index=torch.argsort(dist,dim=1)[:,0]

                pc_sel=pc_sel[:,sel_index].detach().cpu().numpy()
                img_xy_pc=img_xy_flatten_sel.detach().cpu().numpy()

                is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                            iterationsCount=500,
                                                            reprojectionError=args.dist_thres,
                                                            flags=cv2.SOLVEPNP_EPNP,
                                                            distCoeffs=None)

                R,_=cv2.Rodrigues(R)
                T_pred=np.eye(4)
                T_pred[0:3,0:3]=R
                T_pred[0:3,3:]=t
                t_diff,angles_diff=get_P_diff(T_pred,P)
                t_diff_set.append(t_diff)
                angles_diff_set.append(angles_diff)

            if step % 100 == 0:
                t_diff_set_np = np.array(t_diff_set)
                angles_diff_set_np = np.array(angles_diff_set)

                index=(angles_diff_set_np<5)&(t_diff_set_np<2)
                print('step:', step, '---', 'Good rate : ', t_diff_set_np[index].shape, '/', t_diff_set_np.shape)
                print('RTE mean',np.mean(t_diff_set_np), 'std', np.std(t_diff_set_np))
                print('RRE mean',np.mean(angles_diff_set_np), 'std', np.std(angles_diff_set_np))

    print('------------------Final Results------------------')
    t_diff_set = np.array(t_diff_set)
    angles_diff_set = np.array(angles_diff_set)

    index=(angles_diff_set<5)&(t_diff_set<2)
    print('Good rate : ', t_diff_set[index].shape, '/', t_diff_set.shape)
    print('RTE mean',np.mean(t_diff_set), 'std', np.std(t_diff_set))
    print('RRE mean',np.mean(angles_diff_set), 'std', np.std(angles_diff_set))

    return t_diff_set, angles_diff_set

def vis_registration(model,test_dataset,args):
    t_diff_set=np.zeros(len(test_dataset))
    angles_diff_set=np.zeros(len(test_dataset))

    out_dir = os.path.join(args.save_path, args.exp_name) + '/'
    os.makedirs(out_dir, exist_ok=True)

    for i in range(20):
        ind = random.randint(0, len(test_dataset))
        data = test_dataset[ind]   # 2500

        model.eval()
        img=data['img'].cuda().unsqueeze(0)              #full size
        pc_all=data['pc'].cuda().unsqueeze(0)
        intensity=data['intensity'].cuda().unsqueeze(0)
        sn=data['sn'].cuda().unsqueeze(0)
        K_all=data['K'].cuda().unsqueeze(0)
        P_all=data['P'].cuda().unsqueeze(0)
        pc_mask=data['pc_mask'].cuda().unsqueeze(0)      
        img_mask=data['img_mask'].cuda().unsqueeze(0)    #1/4 size

        pc_kpt_idx=data['pc_kpt_idx'].cuda().unsqueeze(0)                #(B,512)
        pc_outline_idx=data['pc_outline_idx'].cuda().unsqueeze(0)
        img_kpt_idx=data['img_kpt_idx'].cuda().unsqueeze(0)
        img_outline_idx=data['img_outline_index'].cuda().unsqueeze(0)
        node_a=data['node_a'].cuda().unsqueeze(0)
        node_b=data['node_b'].cuda().unsqueeze(0)

        img_features_all,pc_features_all,img_score_all,pc_score_all=model(pc_all,intensity,sn,img)     #64 channels feature
        
        bs = img_score_all.shape[0]

        img_score=img_score_all[0]
        pc_score=pc_score_all[0]
        img_feature=img_features_all[0]
        pc_feature=pc_features_all[0]
        pc=pc_all[0]
        P=P_all[0].data.cpu().numpy()
        K=K_all[0].data.cpu().numpy()
        pc_vis = pc.transpose(1,0).detach().cpu().numpy()

        pc_vis = pc_vis.transpose(1,0)
        pc_np_homo = np.concatenate((pc_vis, np.ones((1, pc_vis.shape[1]))), axis=0)  # 4xN
        pc_recovered_all = np.dot(P[:3,:], pc_np_homo).copy() 

        
        # ---------------------------get image corrs------------------------
        img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
        img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

        img_xy=np.concatenate((img_x,img_y),axis=0)
        img_xy = torch.tensor(img_xy).cuda()

        img_xy_flatten=img_xy.reshape(2,-1)
        img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
        img_score_flatten=img_score.squeeze().reshape(-1)

        img_index=(img_score_flatten>args.img_thres)

        img_xy_flatten_sel=img_xy_flatten[:,img_index]
        img_feature_flatten_sel=img_feature_flatten[:,img_index]
        img_score_flatten_sel=img_score_flatten[img_index]
        # --------------------------------------------------------------------

        pc_index=(pc_score.squeeze()>args.pc_thres)
        pc_sel=pc[:,pc_index]
        pc_feature_sel=pc_feature[:,pc_index]
        pc_score_sel=pc_score.squeeze()[pc_index]

        pc_sel_np = pc_sel.detach().cpu().numpy()
        pc_np_homo = np.concatenate((pc_sel_np, np.ones((1, pc_sel_np.shape[1]))), axis=0)  # 4xN
        pc_np_recovered_homo = np.dot(P[:3,:], pc_np_homo).copy() 

        # ------------------------------------------------------------------------------

        dist=1-torch.sum(img_feature_flatten_sel.unsqueeze(2)*pc_feature_sel.unsqueeze(1), dim=0)

        sel_index=torch.argsort(dist,dim=1)[:,0]

        pc_sel=pc_sel[:,sel_index].detach().cpu().numpy()
        img_xy_pc=img_xy_flatten_sel.detach().cpu().numpy()

        is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                    iterationsCount=500,
                                                    reprojectionError=args.dist_thres,
                                                    flags=cv2.SOLVEPNP_EPNP,
                                                    distCoeffs=None)

        R,_=cv2.Rodrigues(R)
        T_pred=np.eye(4)
        T_pred[0:3,0:3]=R
        T_pred[0:3,3:]=t
        t_diff,angles_diff=get_P_diff(T_pred,P)

        t_diff_set[ind] = t_diff
        angles_diff_set[ind] = angles_diff

        # ------------------------------------------------------------------------------

        img_vis = np.copy(img[0].detach().cpu().numpy().transpose(1,2,0))
        H, W = img_vis.shape[0], img_vis.shape[1]

        pc_np_homo = np.concatenate((pc.detach().cpu().numpy(), np.ones((1, pc.shape[1]))), axis=0)  # 4xN
        pc_vis = np.dot(T_pred[:3,:], pc_np_homo).copy() 
        pc_np_front = pc_vis[:, pc_vis[2, :]>0]  # 3xN
        color_dists = np.sqrt(np.power(pc_np_front[0],2) + np.power(pc_np_front[1],2) + np.power(pc_np_front[2],2))

        K_ori = 4 * K
        K_ori[2, 2] = 1
        pc_pixels = np.dot(K_ori, pc_np_front)  # 3xN
        pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
        mid_factor = 0.3

        cm1 = plt.cm.get_cmap('rainbow') # viridis Spectral jet tab20b gist_rainbow

        x = pc_pixels[0,:]
        y = pc_pixels[1,:]
        c = color_dists / 70
        c = np.minimum(c, 1)
        plt.figure()

        sc1=plt.scatter(x, y, s=0.15, c=c,cmap=cm1)

        pc_np_vis = img_vis
        plt.axis('off')
        plt.imshow(pc_np_vis)
        os.makedirs(os.path.join(out_dir, str(i)), exist_ok=True)
        plt.savefig(out_dir + '/' + str(i) + '/' + 'proj_pred', dpi=200, bbox_inches='tight', pad_inches=0.0)

        # ------------------------------------------------------------------------------

        img_vis = np.copy(img[0].detach().cpu().numpy().transpose(1,2,0))
        H, W = img_vis.shape[0], img_vis.shape[1]

        pc_np_front = pc_recovered_all[:, pc_recovered_all[2, :]>1.0]  # 3xN
        color_dists = np.sqrt(np.power(pc_np_front[0],2) + np.power(pc_np_front[1],2) + np.power(pc_np_front[2],2))

        K_ori = 4 * K
        K_ori[2, 2] = 1
        pc_pixels = np.dot(K_ori, pc_np_front)  # 3xN
        pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
        mid_factor = 0.3

        cm1 = plt.cm.get_cmap('rainbow') # viridis Spectral jet tab20b gist_rainbow

        x = pc_pixels[0,:]
        y = pc_pixels[1,:]
        c = color_dists / 70
        c = np.minimum(c, 1)
        plt.figure()

        sc1=plt.scatter(x, y, s=0.15, c=c,cmap=cm1)

        pc_np_vis = img_vis
        plt.axis('off')
        plt.imshow(pc_np_vis)
        plt.savefig(out_dir + '/' + str(i) + '/' + 'proj_gt', dpi=200, bbox_inches='tight', pad_inches=0.0)

        # ------------------------------------------------------------------------------


        img_vis = np.copy(img[0].detach().cpu().numpy().transpose(1,2,0))
        H, W = img_vis.shape[0], img_vis.shape[1]

        pc_ori = pc_recovered_all
        pc_vis = pc.detach().cpu().numpy()
        color_dists = np.sqrt(np.power(pc_ori[0],2) + np.power(pc_ori[1],2) + np.power(pc_ori[2],2))

        index = pc_vis[2, :]>1.0
        pc_np_front = pc_vis[:, index]
        color_dists = color_dists[index]

        K_ori = 4 * K
        K_ori[2, 2] = 1
        pc_pixels = np.dot(K_ori, pc_np_front)  # 3xN
        pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
        mid_factor = 0.3

        cm1 = plt.cm.get_cmap('rainbow')

        x = pc_pixels[0,:]
        y = pc_pixels[1,:]
        c = color_dists / 70
        c = np.minimum(c, 1)
        plt.figure()

        sc1=plt.scatter(x, y, s=0.15, c=c,cmap=cm1)

        pc_np_vis = img_vis
        plt.axis('off')
        plt.imshow(pc_np_vis)
        plt.savefig(out_dir + '/' + str(i) + '/' + 'proj_input', dpi=200, bbox_inches='tight', pad_inches=0.0)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=25, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=12, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=2, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='/data/kitti_testset/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--input_pt_num', type=int, default=40960, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=1000, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='')
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2*math.pi*0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2*math.pi*0, metavar='P_Rz_amplitude',
                        help='')

    parser.add_argument('--save_path', type=str, default='./outs', metavar='save_path',
                        help='path to save log and model')

    parser.add_argument('--exp_name', type=str, default='test', metavar='save_path',
                    help='path to save log and model')

    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',    # save radius
                        help='')    

    parser.add_argument('--img_thres', type=float, default=0.95, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.95, metavar='pc_thres',
                        help='')

    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')
    parser.add_argument('--load_ckpt', type=str, default='none', metavar='save_path',
                    help='path to save log and model')

    parser.add_argument('--mode', type=str, default='none', metavar='save_path',
                    help='path to save log and model')
    
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logdir=os.path.join(args.save_path, args.exp_name)
    try:
        os.makedirs(logdir)
    except:
        print('mkdir failue')

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (logdir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    opt=options.Options()
    opt.input_pt_num = args.input_pt_num
    
    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.input_pt_num,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)

    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    model=VP2PMatchNet(opt)

    model=model.cuda()
    if args.load_ckpt != 'none':
        model.load_state_dict(torch.load(args.load_ckpt)) 

    if args.mode == 'vis_registration':
        vis_registration(model,test_dataset,args)

    else:
        t_diff,r_diff=test_acc_trans(model,testloader,args)
