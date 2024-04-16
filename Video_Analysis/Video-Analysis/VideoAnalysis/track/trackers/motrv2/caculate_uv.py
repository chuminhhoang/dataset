import numpy as np
import os

def getUVError(box):
    u = 0.05*box[3]
    v = 0.05*box[3]
    if u>13:
        u = 13
    elif u<2:
        u = 2
    if v>10:
        v = 10
    elif v<2:
        v = 2
    return u,v
    

def parseToMatrix(data, rows, cols):
    matrix_data = np.fromstring(data, sep=' ')
    matrix_data = matrix_data.reshape((rows, cols))
    return matrix_data

def readKittiCalib(filename):
    # 检查文件是否存在
    if not os.path.isfile(filename):
        print(f"Calib file could not be opened: {filename}")
        return None,False

    P2 = np.zeros((3, 4))
    R_rect = np.identity(4)
    Tr_velo_cam = np.identity(4)
    KiKo = None

    with open(filename, 'r') as infile:
        for line in infile:
            id, data = line.split(' ', 1)
            if id == "P2:":
                P2 = parseToMatrix(data, 3, 4)
            elif id == "R_rect":
                R_rect[:3, :3] = parseToMatrix(data, 3, 3)
            elif id == "Tr_velo_cam":
                Tr_velo_cam[:3, :4] = parseToMatrix(data, 3, 4)
            KiKo = np.dot(np.dot(P2, R_rect), Tr_velo_cam)

    return KiKo, True

def readCamParaFile(camera_para, camera_para_1, camera_para_2):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    IntrinsicMatrix = np.zeros((3, 3))

    R = camera_para
    T = camera_para_1
    T = T / 1000
    IntrinsicMatrix = camera_para_2

    Ki = np.zeros((3, 4))
    Ki[:, :3] = IntrinsicMatrix

    Ko = np.eye(4)
    Ko[:3, :3] = R
    Ko[:3, 3] = T.flatten()

    KiKo = np.dot(Ki, Ko)
    return KiKo

class Mapper(object):
    def __init__(self, campara_param, campara_param_1, campara_param_2,dataset= "kitti"):
        self.A = np.zeros((3, 3))
        if dataset == "kitti":
            # self.KiKo, self.is_ok = readKittiCalib(campara_file)
            self.KiKo= readCamParaFile(campara_param, campara_param_1, campara_param_2)
            self.is_ok = True
            z0 = -1.73
        else:
            # self.KiKo, self.is_ok = readCamParaFile(campara_file)
            self.KiKo = readCamParaFile(campara_param, campara_param_1, campara_param_2)
            self.is_ok = True
            z0 = 0
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = z0 * self.KiKo[:, 2] + self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def uv2xy(self, uv, sigma_uv):
        if self.is_ok == False:
            return None, None

        uv1 = np.zeros((3, 1))
        uv1[:2,:] = uv
        uv1[2,:] = 1
        b = np.dot(self.InvA, uv1)
        gamma = 1 / b[2,:]
        C = gamma * self.InvA[:2, :2] - (gamma**2) * b[:2,:] * self.InvA[2, :2]
        xy = b[:2,:] * gamma
        sigma_xy = np.dot(np.dot(C, sigma_uv), C.T)
        return xy, sigma_xy
    
    def xy2uv(self,x,y):
        if self.is_ok == False:
            return None, None
        xy1 = np.zeros((3, 1))
        xy1[0,0] = x
        xy1[1,0] = y
        xy1[2,0] = 1
        uv1 = np.dot(self.A, xy1)
        return uv1[0,0]/uv1[2,0],uv1[1,0]/uv1[2,0]
    
    def mapto(self,box):
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        y,R = self.uv2xy(uv, sigma_uv)
        return y,R
