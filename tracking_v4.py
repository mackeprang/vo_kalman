import multiprocessing as mp
import init_auv as auv
import cv2
import time
import tracker_v2 as tr
import kalman_v2 as kf
import numpy as np
from scipy import integrate
import math
import csv

MISSION_NUMBER = int(15)
mission = auv.get_mission_by_num(MISSION_NUMBER)
start_idx = 0
goal_idx = -1
filenames = auv.find_images(mission["Images"])
filenames = filenames[start_idx:goal_idx]
f = auv.read_h5(mission["Data"])
data_time = f["Position"]["time"]
data_time = data_time[start_idx:goal_idx]
vo_time = auv.match_time_stamps(filenames,list(data_time))
Time = data_time - data_time[0]
Time = Time / 1000

def preProcessImg(q1,running):
    first = True
    print("preProcess started...")
   #while running.value==1:
    for frame in filenames:
        if running.value == 0:
            print("imgProcess ended...")
            return
        if auv.image_broken(frame):
            continue
        im = cv2.imread(frame)
        gray = auv.preprocess_image(im, size=auv.IM_PRE_RESOLUTION,method=auv.IM_PRE_EQ_HIST)

        q1.put(gray)


def voProcessFunc(image_queue, CposQueue,scale,running):
    feat = tr.featureDetector('good')
    tracker = tr.tracking(feat)
    print("voProcess started...")
    while True:
        t1 = time.time()
        if running.value==0:
            time.sleep(1)
            print("voProcess ended...")
            return

        while image_queue.empty():
            time.sleep(0.01)

        im = image_queue.get()
        
        tracker.track(im,verbose=True)
        #print(tracker._C)
        CposQueue.put({'R':tracker._R,'t':tracker._t,'scale':tracker._scale[-1],'fps':1/(time.time()-t1)})
        scale.value = tracker._scale[-1]
        #print("FPS: {:.3f}".format(1/(time.time()-t1)))

        

def plotterProcess(CposQueue,scale,running):
    import matplotlib.pyplot as plt
    import numpy as np
    import init_auv as auv
    from mpl_toolkits.mplot3d import Axes3D  # Do not remove this line
    gps_x = auv.getRelPos(mission["Data"])["X"]
    gps_y = auv.getRelPos(mission["Data"])["Y"]
    C = None
    plt.figure(figsize=(10, 5))
    plt.axis('equal')
    plt.plot(gps_x, gps_y, '.')
    plt.ion()
    plt.show(False)
    print("Plotter starting...")
    while True:
        if running.value == 0:
            print("Plotter ended...")
            return

        C_CAM = CposQueue.get()
        if C is None:
            print("Here")
            C = np.dot(kf.C_CAM_to_AUV,C_CAM)
            continue
        C_CAM = np.dot(kf.C_CAM_to_AUV,C_CAM)
        #C_CAM = np.multiply(C_CAM,kf.getTransformationMat(np.ones((3,3)),np.array([[scale.value],[scale.value],[scale.value]])))
        C = np.dot(C,C_CAM)
        plt.plot(C[0,3], C[1,3], '.', color='r')
        plt.pause(0.01)

def kalmanProcess(CposQueue,kfPosQueue,scale_cam,running):

    accX = f["Position"]["Acc_X"]
    accY = f["Position"]["Acc_Y"]
    accZ = f["Position"]["Acc_Z"]
    rollRate = f["Position"]["Roll_Rate"]
    pitchRate = f["Position"]["Pitch_Rate"]
    headingRate = f["Position"]["Heading_Rate"]
    depth = f["Position"]["Depth"]

    accX = accX[start_idx:goal_idx]
    accY = accY[start_idx:goal_idx]
    accZ = accZ[start_idx:goal_idx]
    rollRate = rollRate[start_idx:goal_idx]
    pitchRate = pitchRate[start_idx:goal_idx]
    headingRate = headingRate[start_idx:goal_idx]
    depth = depth[start_idx:goal_idx]
    data_vars = open("data_variance.txt")
    reader = csv.reader(data_vars)
    next(reader)
    variance = next(reader)
    data_vars.close()

    accXVariance = float(variance[0])
    accYVariance = float(variance[1])
    accZVariance = float(variance[2])
    gyroRollVariance = float(variance[3])
    gyroPitchVariance = float(variance[4])
    gyroHeadingVariance = float(variance[5])
    depth_variance = float(variance[6])

    ### Normal distributed depth for test data ###
    depth = np.random.normal(1.5, depth_variance, accX.shape)

    rollFromGravity = np.zeros(len(accX))
    pitchFromGravity = np.zeros(len(accY))

    ### REMOVE GRAVITY WITH LOW PASS FILTER ###
    alpha = 0.8
    gravityX = np.zeros((len(accX)))
    gravityY = np.zeros((len(accY)))
    gravityZ = np.zeros((len(accZ)))
    gravityX[0] = accX[0]
    gravityY[0] = accY[0]
    gravityZ[0] = accZ[0]

    for i in range(len(accX)):
        if i == 0:
            continue
        gravityX[i] = gravityX[i - 1] * alpha + (1 - alpha) * accX[i]
        gravityY[i] = gravityY[i - 1] * alpha + (1 - alpha) * accY[i]
        gravityZ[i] = gravityZ[i - 1] * alpha + (1 - alpha) * accZ[i]
        rollFromGravity[i] = math.atan2(accZ[i], accY[i]) * 180 / math.pi  # atan2(accY, accZ)
        pitchFromGravity[i] = math.atan2(accX[i], (
            math.sqrt(accZ[i] * accZ[i] + accY[i] * accY[i]))) * 180 / math.pi  # atan2(-accX, sqrt(accY^2+accZ^2))

    # accX = accX - gravityX
    # accY = accY - gravityY
    # accZ = accZ - gravityZ
    AccX_Value = accX - gravityX  # -(accX - gravityX)
    AccY_Value = accY - gravityY  # accZ - gravityZ
    AccZ_Value = accZ - gravityZ  # accY - gravityY

    # rollRate = -rollRate
    # dummy = pitchRate
    # pitchRate = headingRate
    # headingRate = dummy

    velX = integrate.cumtrapz(AccX_Value, Time)
    posX = integrate.cumtrapz(velX, Time[1:])
    velY = integrate.cumtrapz(AccY_Value, Time)
    posY = integrate.cumtrapz(velY, Time[1:])
    velZ = integrate.cumtrapz(AccZ_Value, Time)
    posZ = integrate.cumtrapz(velZ, Time[1:])
    yaw = integrate.cumtrapz(headingRate, Time)
    dt = 0.1
    scale = []
    for j in range(len(posX)):
        scale.append(
            math.sqrt((posX[j] - posX[j - 1]) ** 2 + (posY[j] - posY[j - 1]) ** 2 + (posZ[j] - posZ[j - 1]) ** 2))

    mscale = np.mean(scale)

    # transition_matrix (A)
    F = kf.transitionMat()
    # observation_matrix (C)

    H = kf.observationMat()

    # transition_covariance
    Q = np.diag((0.2, 0.1, 0.001, 0.2, 0.01, 0.001, 0.2, 0.1, 0.001, 0.2, 0.001, 0.2, 0.001, 0.2, 0.001, 1))

    # observation_covariance
    R = np.diag((accXVariance, accYVariance, accZVariance, depth_variance, gyroRollVariance, gyroPitchVariance,
                 gyroHeadingVariance, 10, 10, 1, 1))
    # initial_state_mean
    X0 = np.array([
        0,  # 0:  pos x
        0,  # 1:  vel x
        AccX_Value[0],  # 2:  acc x
        0,  # 3:  pos y
        0,  # 4:  vel y
        AccY_Value[0],  # 5:  acc y
        -depth[0],  # 6:  pos z
        0,  # 7:  vel z
        AccZ_Value[0],  # 8:  acc z
        0,  # 9:  roll
        rollRate[0],  # 10: roll rate
        0,  # 11: pitch
        pitchRate[0],  # 12: pitch rate
        0,  # 13: yaw
        headingRate[0],  # 14: yaw rate
        0])  # 15: scale

    # initial_state_covariance
    P0 = np.diag((0, 0, accXVariance, 0, 0, accYVariance, 0, 0, accZVariance, 0, gyroRollVariance, 0, gyroPitchVariance,
                  0, gyroHeadingVariance, 0.1))

    n_timesteps = len(vo_time)
    n_dim_state = 16
    fsm = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    t_cam = np.zeros((3,1),dtype=np.float32)
    tposKf = np.zeros((n_timesteps, 3))
    tdiff = np.zeros((3,1))
    i = -1
    gps_travel_dist = 0.0
    kf_travel_dist = 0.0
    print("Kalman filter starting...")
    for t, vo_t in enumerate(vo_time):
        t_start = time.time()
        if running.value == 0:
            print("Closing Kalman filter...")
            return
        if vo_t == 2:  # Visuel odometry
            pass
            t_cam = CposQueue.get()
            t_cam = t_cam[0:3,3]
            #print(t_cam)
        if i > -1:
            try:
                scaleIMU = scale[t]
            except:
                scaleIMU = scale[-1]
            acc_w = np.dot(kf.Rot_IMU_to_world, np.array(
                [
                    [AccX_Value[t]],
                    [AccY_Value[t]],
                    [AccZ_Value[t]]
                ]))
            tpos_w = kf.tpos_CAM_to_world + np.dot(kf.Rot_CAM_to_world, t_cam)
            angular_w = np.dot(kf.Rot_IMU_to_world, np.array([[rollRate[t]],
                                                              [pitchRate[t]],
                                                              [headingRate[t]]]))
            z = [acc_w[0],
                 acc_w[1],
                 acc_w[2],
                 depth[t],
                 angular_w[0],
                 angular_w[1],
                 angular_w[2],
                 tpos_w[0],
                 tpos_w[1],
                 scale_cam.value,
                 scaleIMU]

            if t == 0:
                fsm[t] = X0  # Filtered State Means
                filtered_state_covariances[t] = P0
                continue
            else:
                fsm[t], filtered_state_covariances[t] = (
                    kf.predict(fsm[t - 1],
                               filtered_state_covariances[t - 1],
                               F,
                               Q)
                )
                fsm[t], filtered_state_covariances[t], _ = (
                    kf.update(fsm[t],
                              filtered_state_covariances[t],
                              z,
                              H,
                              R)
                )
            if t > 0:
                tdiff = np.array([[fsm[t, 0] - fsm[t-1, 0]],
                     [fsm[t, 3] - fsm[t-1, 3]],
                     [fsm[t, 6] - fsm[t-1, 6]]])

        xAngle = math.radians(rollFromGravity[t])
        yAngle = math.radians(pitchFromGravity[t])
        zAngle = math.radians(fsm[t, 13])

        # rot = auv.calculate_rotation_matrix(deg2rad(fsm[t,9]),deg2rad(fsm[t,11]),deg2rad(fsm[t,13]))
        rot = auv.calculate_rotation_matrix(xAngle, yAngle, zAngle)

        tposKf[t] = tposKf[t-1] + np.transpose(np.dot(rot, tdiff)) * fsm[t, -1] # scale
        #print("Kalman: {:.3f}".format(1/(time.time()-t_start)))


if __name__ == '__main__':
    process = []
    q = []
    mp.set_start_method('fork')
    scale_cam = mp.Value('d',0.2)
    lock_scale_cam = mp.Lock()
    scale_imu = mp.Value('d',0.2)
    lock_scale_imu = mp.Lock()
    running = mp.Value('i',1)
    preProcessImgQueue1 = mp.Queue()
    kfPosQueue = mp.Queue()
    voQueue = mp.Queue()
    CposQueue = mp.Queue()
    q.append(preProcessImgQueue1)
    q.append(voQueue)
    q.append(CposQueue)

    imgProcess = mp.Process(target=preProcessImg, args=(preProcessImgQueue1,running))
    voProcess1 = mp.Process(target=voProcessFunc,args=(preProcessImgQueue1,CposQueue,scale_cam,running))
    kalman = mp.Process(target=kalmanProcess,args=(CposQueue,kfPosQueue,scale_cam,running))
    #plotter = mp.Process(target=plotterProcess,args=(CposQueue,kfPosQueue,scale_cam,running))
    process.append(imgProcess)
    process.append(voProcess1)
    #process.append(plotter)
    process.append(kalman)
    for p in process:
        p.start()
    for t in range(100):
        time.sleep(1)
        voT = CposQueue.get()
        print("fps: ",voT['fps'])
    running.value = 0


    for p in process:
        if p.join(timeout=5):
            print("here")
        p.terminate()

    print("Program ended...")
    exit()
