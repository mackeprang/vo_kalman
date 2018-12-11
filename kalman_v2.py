import numpy as np
import math

def getTransformationMat(R,t):
    T = np.array([
                  [R[0, 0], R[0, 1], R[0, 2], t[0]],
                  [R[1, 0], R[1, 1], R[1, 2], t[1]],
                  [R[2, 0], R[2, 1], R[2, 2], t[2]],
                  [0      ,    0   ,    0   ,   1 ]
                  ])
    return T

def deg2rad(deg):
    return deg*np.pi/180

def rad2deg(rad):
    return rad * 180 / np.pi

def calculate_rotation_matrix(x,y,z,deg=True):
    if deg:
        x = deg2rad(x)
        y = deg2rad(y)
        z = deg2rad(z)
    theta = [x, y, z]
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    R[abs(R)<0.000001] = 0.0

    return R


Rot_IMU_to_world = calculate_rotation_matrix(90,0,180) # OK !!! Tested in Matlab
Rot_CAM_to_world = calculate_rotation_matrix(0,0,-90+68)
Rot_world_to_GPS = calculate_rotation_matrix(0,0,205)
# Rot_IMU_to_world = calculate_rotation_matrix(90,180,0)
# Rot_IMU_to_world = calculate_rotation_matrix(0,180,0) #
tpos_IMU_to_world = np.array([[0],[0],[0]],dtype=np.float32)
tpos_CAM_to_world = np.array([[-0.181],[0],[0.0273]],dtype=np.float32)
tpos_world_to_GPS = np.array([[0],[0],[0]],dtype=np.float32)

C_IMU_to_AUV = getTransformationMat(Rot_IMU_to_world,tpos_IMU_to_world)
C_CAM_to_AUV = getTransformationMat(Rot_CAM_to_world,tpos_CAM_to_world)

depth_offset = 0

def predict(X, P, A, Q):
    X = np.dot(A, X)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return(X, P)

def update(X, P, Y, C, R):
    IS = np.dot(C, np.dot(P, C.T)) + R
    K = np.dot(P, np.dot(C.T, np.linalg.inv(IS)))
    X = X + np.dot(K, (Y - np.dot(C, X)))
    P = P - np.dot(K, np.dot(IS, K.T))
    return(X, P, K)

def transitionMat(rot=calculate_rotation_matrix(0,0,0),dt=0.1,scale=0.158):
    return np.array([
        [rot[0,0], dt/scale, (0.5*dt**2)/scale, rot[0,1],  0,         0, rot[0,2],  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [0,  1,        dt, 0,  0,         0, 0,  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         1, 0,  0,         0, 0,  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [rot[1,0],  0,         0, rot[1,1], dt/scale, (0.5*dt**2)/scale, rot[1,2],  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  1,        dt, 0,  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         1, 0,  0,         0,  0, 0,  0, 0,  0, 0, 0],
             [rot[2,0],  0,         0, rot[2,1],  0,         0, rot[2,2], dt/scale, (0.5*dt**2)/scale,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  1,        dt,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         1,  0, 0,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  1, dt, 0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 1,  0, 0,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 0,  1, dt, 0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 0,  0, 1,  0, 0, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 0,  0, 0, 1, dt, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 0,  0, 0,  0, 1, 0],
             [0,  0,         0, 0,  0,         0, 0,  0,         0,  0, 0,  0, 0,  0, 0, 1]]
             )

def observationMat():
#        0  1  2  3  4  5  6   7  8  9  10 11 12 13 14 15
    return np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # acc x (2)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # axx y (5)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # acc z (9)
        [0, 0, 0, 0, 0, 0, -1+depth_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # depth (6)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # roll rate (10)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # pitch rate (12)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # heading rate (14)
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # tracker._tTrue (0)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # tracker._tTrue (3)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # scale vo (15)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # scale imu (15)
    ]
    )

