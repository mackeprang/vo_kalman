import numpy as np
import init_auv as auv
import math
#import matplotlib.pyplot as plt
import tracker_v2 as tr
import cv2
import csv
import time
from scipy import integrate
import kalman_v2 as kf

MISSION_NUMBER = int(15)

#from mpl_toolkits.mplot3d import Axes3D # Do not remove this line


def deg2rad(deg):
    return deg*np.pi/180

def rad2deg(rad):
    return rad * 180 / np.pi



qx = None
def draw(ax,ax2,Rpos,tpos):
    global qx, qy, qz, qg, accX, accY,accZ
    tpos = np.dot(kf.Rot_world_to_GPS,tpos)
    Rpos = np.dot(kf.Rot_world_to_GPS,Rpos)
    acc_g = np.dot(kf.Rot_world_to_GPS,np.dot(kf.Rot_IMU_to_world,np.array([[accX[t]],
                                                 [accY[t]],
                                                 [accZ[t]]
                                                 ],dtype=np.float32)))
    gravityVector = -np.array([acc_g[0], [acc_g[1]], [acc_g[2]]]) / math.sqrt(acc_g[0] ** 2 + acc_g[1] ** 2 + acc_g[2] ** 2)
    if qx:
        ax.collections.remove(qx)
        ax.collections.remove(qy)
        ax.collections.remove(qz)
        ax.collections.remove(qg)
    line = ax2.plot(tpos[0], tpos[1], '.', color='r')
    qx = ax.quiver(0, 0, 0, Rpos[0, 0], Rpos[1, 0], Rpos[2, 0], color=['b'])
    qy = ax.quiver(0, 0, 0, Rpos[0, 1], Rpos[1, 1], Rpos[2, 1], color=['r'])
    qz = ax.quiver(0, 0, 0, Rpos[0, 2], Rpos[1, 2], Rpos[2, 2], color=['g'])
    qg = ax.quiver(0, 0, 0, gravityVector[0], gravityVector[1], gravityVector[2], color=['k'])
    ax.legend(["X", "Y", "Z", "G"])
    # ax.legend(["X", "Y", "Z"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    plt.draw()

mission = auv.get_mission_by_num(MISSION_NUMBER)
# Rot_world_to_GPS = auv.calculate_rotation_matrix(0,0,250,deg=True) # 250 is a guess of the auv's heading
feat = tr.featureDetector("good")
tracker = tr.tracking(feat)
# tracker._Rpos = auv.calculate_rotation_matrix(0,0,0,deg=True) # 250 is a guess of the auv's heading

f = auv.read_h5(mission["Data"])
accX = f["Position"]["Acc_X"]
accY = f["Position"]["Acc_Y"]
accZ = f["Position"]["Acc_Z"]

gps_x = auv.getRelPos(mission["Data"])["X"]
gps_y = auv.getRelPos(mission["Data"])["Y"]

start_idx = 0
goal_idx = -1

data_time = f["Position"]["time"]
data_time = data_time[start_idx:goal_idx]
Time = data_time - data_time[0]
Time = Time / 1000

filenames = auv.find_images(mission["Images"])


# delta_time = auv.get_timestamps(filenames,diff=True,scale=1)
# delta_time = delta_time[start_idx:goal_idx]
filenames = filenames[start_idx:goal_idx]
vo_time = auv.match_time_stamps(filenames,list(data_time))

#### ---- Kalman settings ---- ####
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
    gravityX[i] = gravityX[i-1] * alpha + (1 - alpha) * accX[i]
    gravityY[i] = gravityY[i-1] * alpha + (1 - alpha) * accY[i]
    gravityZ[i] = gravityZ[i-1] * alpha + (1 - alpha) * accZ[i]
    rollFromGravity[i] = math.atan2(accZ[i], accY[i]) * 180/math.pi # atan2(accY, accZ)
    pitchFromGravity[i] = math.atan2(accX[i], (math.sqrt(accZ[i]*accZ[i] + accY[i]*accY[i])))*180/math.pi # atan2(-accX, sqrt(accY^2+accZ^2))

# accX = accX - gravityX
# accY = accY - gravityY
# accZ = accZ - gravityZ
AccX_Value = accX - gravityX#-(accX - gravityX)
AccY_Value = accY - gravityY#accZ - gravityZ
AccZ_Value = accZ - gravityZ#accY - gravityY

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
    scale.append(math.sqrt((posX[j]-posX[j-1])**2 + (posY[j]-posY[j-1])**2 + (posZ[j]-posZ[j-1])**2))

mscale = np.mean(scale)


# transition_matrix (A)
F = kf.transitionMat()
# observation_matrix (C)

H = kf.observationMat()

# transition_covariance
Q = np.diag((0.2, 0.1, 0.001, 0.2, 0.01, 0.001, 0.2, 0.1, 0.001, 0.2, 0.001, 0.2, 0.001, 0.2, 0.001, 1))

# observation_covariance
R = np.diag((accXVariance, accYVariance, accZVariance, depth_variance, gyroRollVariance, gyroPitchVariance, gyroHeadingVariance, 10, 10, 1, 1))
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
                  0]) # 15: scale

# initial_state_covariance
P0 = np.diag((0, 0, accXVariance, 0, 0, accYVariance, 0, 0, accZVariance, 0, gyroRollVariance, 0, gyroPitchVariance, 0, gyroHeadingVariance, 0.1))

n_timesteps = len(vo_time)
n_dim_state = 16
fsm = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

#### ---- Plot settings ---- ####
#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(121, projection='3d')
#ax2 = fig.add_subplot(122)
#ax2.axis('equal')
#ax2.set_xlim([-25, 5])
#ax2.set_ylim([-15, 10])
#gps_line = ax2.plot(gps_x,gps_y,'.')

#plt.ion()
#plt.show(False)
#print(len(scale))
#################### MAIN LOOP #######################
tposKf = np.zeros((n_timesteps, 3))
i = -1
gps_travel_dist = 0.0
kf_travel_dist = 0.0

for t, vo_t in enumerate(vo_time):

    # if t>1:
    #     F = transitionMat(Rpos,0.1,0.158)
    if vo_t==1: # Visuel odometry
        start_time = time.time()
        i+=1
        # if i % 2 != 0:
        #     continue
        # if i == 564:
        #     print("image time: ",filenames[t])
        #     print("data time: ",data_time[t])
        #     print("t: ",t)
        #     exit()

        frame = filenames[i]
        if auv.image_broken(frame):
            updateVO = False
        else:
            updateVO = True

        if updateVO:
            im = cv2.imread(frame)
            # rows, cols,_ = im.shape
            # M = cv2.getRotationMatrix2D((cols/2,rows/2),-68,1)
            # im = cv2.warpAffine(im, M, (cols, rows))
            gray = auv.preprocess_image(im, size=None,method=auv.IM_PRE_EQ_HIST)
            tracker.track(gray)

        # print("FPS: {:.2f}".format(1.0 / (time.time() - start_time)))
    if i>-1:
        try:
            scaleIMU = scale[t]
        except:
            scaleIMU = scale[-1]
        acc_w = np.dot(kf.Rot_IMU_to_world,np.array(
            [
            [AccX_Value[t]],
            [AccY_Value[t]],
            [AccZ_Value[t]]
            ]))
        tpos_w = kf.tpos_CAM_to_world+np.dot(kf.Rot_CAM_to_world,tracker._tTrue)
        angular_w = np.dot(kf.Rot_IMU_to_world,np.array([[rollRate[t]],
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
             tracker._scale[-1],
             scaleIMU]

        if t == 0:
            fsm[t] = X0 #Filtered State Means
            filtered_state_covariances[t] = P0
            continue
        else:
            fsm[t], filtered_state_covariances[t] = (
                kf.predict(fsm[t-1],
                filtered_state_covariances[t-1],
                F,
                Q)
            )
            fsm[t], filtered_state_covariances[t],_ = (
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

        cv2.imshow("frame",im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): #Pause
            key = cv2.waitKey()
            if key & 0xFF == ord('q'): #Exiting
                break
        tposKf[t] = tposKf[t-1] + np.transpose(np.dot(rot, tdiff)) * fsm[t, -1] # scale
        # acceration = np.array([[fsm[t,2]],[fsm[t,5]],[fsm[t,8]]])*0.01/(2*fsm[t,-1])
        # velocity = np.array([[fsm[t,1]],[fsm[t,4]],[fsm[t,7]]])*0.1/fsm[t,-1]
        # tposKf[t] = tposKf[t-1] + np.transpose(np.dot(rot, velocity))# + np.transpose(np.dot(rot, acceration))# scale
        #draw(ax,ax2,rot,tposKf[t])

        gps_travel_dist += math.sqrt((gps_y[t]-gps_y[t-1])**2+(gps_x[t]-gps_x[t-1])**2)
        kf_travel_dist +=  math.sqrt((tposKf[t,0]-tposKf[t-1,0])**2+(tposKf[t,1]-tposKf[t-1,1])**2)
        print("Frame {}/{}, no. of newFeat {}".format(i,len(filenames),len(tracker._newPoints)))
        print("Frame {}/{}, Depth: {:.2f}, Scale: {:.3f}, FPS: {:.2f}, No. of nUpdateCount: {}".format(i,len(filenames),fsm[t,6],tracker._scale[-1],1/(time.time()-start_time),tracker._nUpdateCount))
        # tracker._tpos = tposKf[t].copy()
        # print(tracker._tTrue)
        # print("Frame {}: Travel dist: GPS: {:.2f}, KF: {:.2f}, diff: {:.2f}".format(i,gps_travel_dist,kf_travel_dist,abs(gps_travel_dist-kf_travel_dist)))
        tracker._Rpos = np.dot(kf.Rot_CAM_to_world.T,rot.copy())
        # tracker._tTrue = np.array([[fsm[t,0]],[fsm[t,3]],[fsm[t,6]]])
        # if t > 1:
        #     print("tpos_diff x: {:.3f}, vel x: {:.3f}".format((fsm[t, 0] - fsm[t-1,0]),fsm[t,1]*0.1/fsm[t,-1]))
print('Ending program')
cv2.waitKey()
plt.close()
cv2.destroyAllWindows()
