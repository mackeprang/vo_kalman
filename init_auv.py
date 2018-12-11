import numpy as np
import cv2
from PIL import Image
import h5py
import utm
import glob
from CONSTANTS import *
from auv_dataset import *
import math
import os
import datetime
import time
#### Global ####
_firstPos = True

################ GETTING DATA ################

def deg2rad(deg):
    return deg*np.pi/180

def rad2deg(rad):
    return rad * 180 / np.pi

def get_sys_platform(platform=SYS_PLATFORM):
    if platform == "darwin":
        return 0
    elif platform == "win32":
        return 1

def image_broken(img_path):
    try:
        im = Image.open(img_path)
        im.verify()
        im.close()
        return 0
    except:
        return 1

def read_h5(file_path):
    return h5py.File(file_path, 'r')

def getAccData(hdf):
    f = read_h5(hdf)
    acc_x = f["Position"]["Acc_X"]
    acc_y = f["Position"]["Acc_Y"]
    acc_z = f["Position"]["Acc_Z"]

    return {"Acc_X": acc_x,"Acc_Y":acc_y,"Acc_Z":acc_z}

def getRelPos(hdf):
    f = read_h5(hdf)
    lat = f["Position"]["Lat"]
    lon = f["Position"]["Lon"]
    xy = []
    for x, y in zip(lat, lon):
        xy.append(utm.from_latlon(x, y))

    rel_pos_x = []
    rel_pos_y = []
    for pos in xy:
        rel_pos_x.append(pos[0] - xy[0][0])
        rel_pos_y.append(pos[1] - xy[0][1])

    return {"X":rel_pos_x,"Y":rel_pos_y}

def draw_gps_marks(canvas,gps_x,gps_y,size=(480,400),radius=1):
    for i,k in zip(gps_x,gps_y):
        x = int(i + 400//2)
        y = int(-k+ 480//2)
        canvas = cv2.circle(canvas,(x,y),radius,(255,255,255))
    return canvas

def get_gps_goal(gps_pos_x,gps_pos_y,threshold=50):
    idx = 0
    for i, (x, y) in enumerate(zip(gps_pos_x, gps_pos_y)):
        dist = math.sqrt(x * x + y * y)
        if dist > threshold:
            return x,y,i

def get_gps_start(gps_pos_x,gps_pos_y,threshold=50):
    idx = 0
    for i, (x, y) in enumerate(zip(gps_pos_x, gps_pos_y)):
        dist = math.sqrt(x * x + y * y)
        if dist > threshold:
            return gps_pos_x[i-1],gps_pos_y[i-1],i-1

def get_altitude(mission):
    f = read_h5(mission["Data"])
    return f["Position"]["Altitude"]

def getCamMat():
    return np.loadtxt("cam_mat.csv", dtype=np.float32, delimiter=',')

def getDistMat():
    return np.loadtxt("dist_coeff.csv", dtype=np.float32, delimiter=',')

def find_images(imdir,ext='*.png'):
    filenames = []
    filepath = ''.join((imdir, '/', ext))

    for imfile in glob.glob(filepath):
        filenames.append(imfile)
    filenames.sort()
    return filenames

def get_mission_by_num(mission_num=1):
    #if get_sys_platform() == 0:
    #    start_path = '/Users/Mackeprang/Dropbox (Personlig)/'
    #elif get_sys_platform() == 1:
    #    start_path = 'C:/Users/Rasmus/Dropbox/'
    start_path = ''
    imdir,datadir = get_im_and_data_dir(mission_num)
    imdir = ''.join((start_path,imdir))
    datadir = ''.join((start_path,datadir))
    name = ''.join(("Mission ", str(mission_num)))
    return {"Images": imdir,"Data": datadir, "Name": name}

################# OpenCV Functions ##################
def draw_flow(img,p1,p2,mask=None):
    if mask is None:
        p1 = np.reshape(p1,(-1 , 2))
        p2 = np.reshape(p2,(-1 , 2))
        if len(p1) > 0:
            for i, (new,old) in enumerate(zip(p1,p2)):
                a,b = new.ravel()
                c,d = old.ravel()
                img = cv2.line(img,(a,b),(c,d),(0,255,0),1)
    else:
        p1_inliers = p1[mask==1]
        p2_inliers = p2[mask == 1]
        p1_outliers = p1[mask == 0]
        p2_outliers = p2[mask == 0]
        if len(p1_inliers) > 0:
            p1_inliers = np.reshape(p1_inliers,(-1,2))
            p2_inliers = np.reshape(p2_inliers, (-1, 2))
            for i, (new, old) in enumerate(zip(p1_inliers, p2_inliers)):
                a, b = new.ravel()
                c, d = old.ravel()
                img = cv2.arrowedLine(img, (a, b), (c, d), (0, 255, 0), 1,tipLength=0.2)
        if len(p1_outliers) > 0:
            p1_outliers = np.reshape(p1_outliers, (-1, 2))
            p2_outliers = np.reshape(p2_outliers, (-1, 2))
            for i, (new, old) in enumerate(zip(p1_outliers, p2_outliers)):
                a, b = new.ravel()
                c, d = old.ravel()
                img = cv2.arrowedLine(img, (a, b), (c, d), (0, 0, 255), 1,tipLength=0.2)
    return img

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def sparse_optical_flow(img1,img2,points,fb_threshold,optical_flow_params):
    old_points = points.copy()
    new_points, status , err = cv2.calcOpticalFlowPyrLK(img1,img2,points,None,**optical_flow_params)
    if fb_threshold>0:
        new_points_r, status_r, err = cv2.calcOpticalFlowPyrLK(img2,img1,new_points,None,**optical_flow_params)
        new_points_r[status_r==0] = False#np.nan
        fb_good = (np.fabs(new_points_r-points) < fb_threshold).all(axis=2)
        new_points[~fb_good] = np.nan
        old_points = np.reshape(points[~np.isnan(new_points)],(-1,1,2))
        new_points = np.reshape(new_points[~np.isnan(new_points)],(-1,1,2))
    return new_points,old_points

def update_motion(points1,points2,Rpos,tpos,tpos0,cam_mat=None,scale = 1.0, update=True,last_t=None,last_R=None):
    tTrue = tpos0.copy()
    if update:
        try:
            t1 = time.time()
            E, mask = cv2.findEssentialMat(points1,points2,cameraMatrix=cam_mat,method=cv2.RANSAC, mask=None,prob=0.9999)
            t2 = time.time()
        except:
            E = (0,0)
            mask = None
            pass
        if np.shape(E) == (3,3):
            newmask = np.copy(mask)
            t3 = time.time()
            Retval,R,t,newmask = cv2.recoverPose(E,points1,points2,cameraMatrix=cam_mat,mask=newmask)
            t4 = time.time()
        else:
            Retval = 0
        if Retval < 50:
            R = last_R.copy()
            t = last_t.copy()
            Rp = np.dot(last_R, Rpos)
            tp = tpos + np.dot(Rp, last_t) * scale
        else:
            Rp = np.dot(R,Rpos)
            tp = tpos+np.dot(Rp,t)*scale
            tTrue += t
    else:
        R = last_R
        t = last_t
        Rp = np.dot(last_R, Rpos)
        tp = tpos + np.dot(Rp, last_t) * scale
        mask = None
    t5 = time.time()
    # print("findEssen: {:.3f}, recoverPose: {:.3f}, total func: {:.3f}".format(t2-t1,t4-t3,time.time()-t1))
    return Rp, tp, mask, tTrue, t, R

def preprocess_image(im_color,size=IM_PRE_RESOLUTION,method=IM_PRE_EQ_HIST, unsharp=True):
    if size != None:
        im_color = cv2.resize(im_color, size)
    gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
    if unsharp:
        gaussian_im = cv2.GaussianBlur(gray, (9, 9), 10.0)
        gray = cv2.addWeighted(gray, 0.9, gaussian_im, -0.5, 0)
    if method==None:
        res = gray
    elif method==IM_PRE_EQ_HIST:
        res = cv2.equalizeHist(gray)
    elif method == IM_PRE_CLASE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        res = clahe.apply(gray)
    return res

def get_current_heading(Rpos):
    th = math.atan2(Rpos[1][0], Rpos[0][0])
    th_deg = (th*180/3.1415927)
    if th_deg >=360:
        th_deg-=360
    elif th_deg<0:
        th_deg +=360

    return th_deg

def heading_to_euler(heading,rad=True):
    th_deg = 90-heading
    if th_deg < 0 :
        th_deg += 360
    th_rad = th_deg*np.pi/180
    if rad is True:
        return th_rad
    else:
        return th_deg

def draw_path(canvas,points,scale=0.1,clear=True,color=(0,255,0),rotate=1,flipX=False,flipY=False,with_gps_marks=True,gps_x=None,gps_y=None):
    points = np.reshape(points,(-1,3))
    dim = np.shape(canvas)
    canvas_h = dim[0]
    canvas_w = dim[1]
    if clear is True:
        canvas = np.zeros(dim,dtype=np.uint8)
    for n in range(rotate):
        for i, p in enumerate(points):
            points[i]= [-p[1],p[0],p[2]]

    if flipX is True:
        for i, p in enumerate(points):
            points[i]= [-p[0],p[1],p[2]]

    if flipY is True:
        for i, p in enumerate(points):
            points[i]= [p[0],-p[1],p[2]]

    pos_orig = points[0]
    x_orig = pos_orig[0]
    y_orig = pos_orig[1]

    if with_gps_marks:
       canvas = draw_gps_marks(canvas,gps_x,gps_y,radius=2)

    for i, p2 in enumerate(points):
        if i == 0:
            continue
        p1 = points[i-1]
        x1 = int((p1[0]-x_orig)*scale+canvas_w/2)
        y1 = int((p1[1] - y_orig) * scale + canvas_h / 2)
        x2 = int((p2[0] - x_orig) * scale + canvas_w / 2)
        y2 = int((p2[1] - y_orig) * scale + canvas_h / 2)
        cv2.line(canvas,(x1,y1),(x2,y2),color,2)
    return canvas


def calculate_rotation_matrix(x,y,z,deg=False):
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

def triangulate_points(Q1, P1, Q2, P2,threshold=None):
    Q1 = np.reshape(Q1, (-1, 2))
    Q2 = np.reshape(Q2, (-1, 2))
    Q = []
    for (q1, q2) in zip(Q1, Q2):
        B = np.array([P1[2] * q1[0] - P1[0],
                      P1[2] * q1[1] - P1[1],
                      P2[2] * q2[0] - P2[0],
                      P2[2] * q2[1] - P2[1]])

        u, s, v = np.linalg.svd(B, full_matrices=0)
        v = np.transpose(v)
        v = v[:, -1]
        Q = np.hstack((Q, v[:3] / np.repeat(v[3], 3)))

    return Q.reshape(-1, 3)

def get_relative_scale():
    pass

def undistort(points, cm, dc):
    distorted_points = np.array(points, dtype=np.float32)
    distorted_points = np.reshape(distorted_points, (-1, 1, 2))
    undistorted_points = cv2.undistortPoints(distorted_points, cameraMatrix=cm, distCoeffs=dc)
    return undistorted_points


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    #print("Is rotation matrix: " + str(isRotationMatrix(R)))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_timestamps(timestamps,diff=False,scale=1):
    t = []
    t_diff = []
    t_start = []
    for time in timestamps:
        time = time.split('_')
        time = int(''.join((time[-2],time[-1][0:3])))
        if not t_start:
            t_start = time
        t.append((time - t_start)/scale)

    if diff:
        for i,time in enumerate(t):
            if i == 0:
                continue
            t_diff.append((time-t[i-1])/scale)
        t = t_diff
    #print(t)
    return t



def match_time_stamps(date_list,unixtime):

    if not isinstance(unixtime,(list,)):
        unixtime = [unixtime]

    match = np.zeros(len(unixtime))
    match_list = []
    for n,date in enumerate(date_list):
        date = os.path.basename(date)
        if not date[-1].isdigit():
            date = date[:-4]

        date = datetime.datetime.strptime(date, "%Y%m%d_%H%M%S_%f")
        date = int(date.replace(tzinfo=datetime.timezone.utc).timestamp()*1000)
        cand = None
        for i,ut in enumerate(unixtime):
            diff = abs(date-int(ut))
            if cand is None:
                cand = diff
                idx = i
                continue
            if cand > diff:
                cand = diff
                idx = i
                match_list.append((unixtime[i],date))
        match[idx] = 1

    match_list
    return match

def main():
    pass
if __name__ == "__main__":
    main()