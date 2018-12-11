import cv2
import numpy as np
import init_auv as auv
import math
import time
from kalman_v2 import getTransformationMat
class featureDetector:
    def __init__(self,featType):
        self.type = featType
        if featType == "fast":
            self.detector = cv2.FastFeatureDetector_create(**self.getFastParams())
        elif featType == "good":
            pass
        elif featType == "mser":
            self.detector = cv2.MSER_create()
        self.oldPoints = []
        self.newPoints = []

        self.maxFeatures = 500

    def detect(self,im):
        points = []
        keyPoints = []
        if self.type in ("fast","mser"):
            keyPoints = self.detector.detect(im, None)
            for point in keyPoints:
                points.append(point.pt)

            points = np.array(points, dtype=np.float32)
        else:
            points =  cv2.goodFeaturesToTrack(im, mask=None, **self.goodFeatureParams())
            for kp in points:
                kp = kp[0]
                keyPoints.append(cv2.KeyPoint(kp[0], kp[1], _size=2))

        return np.reshape(points, (-1, 1, 2)), keyPoints

    def getFastParams(self):
        return dict(threshold=80,
                    nonmaxSuppression=True,
                    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def goodFeatureParams(self):
        return dict(maxCorners=self.maxFeatures,
                    qualityLevel=0.01,
                    minDistance=5,
                    blockSize=5,
                    useHarrisDetector=True)



class tracking:
    def __init__(self,detector=featureDetector("good"),camMat=auv.getCamMat(),distMat=auv.getDistMat()):
        self._distMat = distMat
        self._camMat = camMat
        self._detector = detector
        self._oldIm = None
        self._newIm = None
        self._points = []
        self._newPoints = []
        self._backward_flow_threshold = 0.8
        self._nUpdateCount = 0
        self._scale = [0]
        self._Rpos = np.eye(3, 3, dtype=np.float32)
        self._tpos = np.zeros((3, 1), dtype=np.float32)
        self._tTrue = np.zeros((3, 1), dtype=np.float32)
        self._C = getTransformationMat(np.eye(3),self._tpos)
        self._t = np.array([[0], [0], [0]])
        self._R = np.eye(3, 3, dtype=np.float32)
        self._VOmask = None
        self._minFeature = 200


    def track(self,gray,verbose=False):
        t1 = time.time()
        if self._oldIm is None:
            self._oldIm = gray.copy()
            self._points, keyPoints = self._detector.detect(gray)
        else:
            self._points, keyPoints = self._detector.detect(gray)
            self._newPoints,self._points = auv.sparse_optical_flow(self._oldIm, gray, self._points, self._backward_flow_threshold,
                                                         self.opticalFlowParams())
            self._oldIm = gray.copy()
            tOF = time.time()
##            try:
##                deltaS = np.zeros(np.shape(self._points.squeeze()),dtype=np.float32)
##                for i, p1 in enumerate(self._points):
##                    p2 = self._newPoints[i]
##                    s = np.subtract(p2,p1).squeeze()
##
##                    m = math.sqrt(s[0]**2+s[1]**2)
##                    deltaS[i] = [s[0]/m , s[1]/m]
##                mdx = np.median(deltaS[:,0],overwrite_input=False)
##                mdy = np.median(deltaS[:,1],overwrite_input=False)
##
##                inliers = np.ones(len(deltaS),dtype=np.uint8)
##                for i, d in enumerate(deltaS):
##                    ddot = d[0]*mdx + d[1]*mdy
##                    if ddot < 0.9:
##                        inliers[i] = 0
##                if np.sum(inliers) > self._minFeature:
##                    self._points = self._points[inliers==1]
##                    self._newPoints = self._newPoints[inliers == 1]
##                # print("median filter (outliers): {}".format(len(deltaS)-np.sum(inliers)))
##            except:
##                # print("no outliers...")
##                pass

            # self._distPoints = self.undistortPoints(self._points)
            # self._distNewPoints = self.undistortPoints(self._newPoints)

            t2 = time.time()
            self.update_motion(update=self.getScale())
            t3  = time.time()
            if verbose:
                print("time OF: {0:.3f}, time median: {2:.3f}, time findEssential: {1:.3f}".format(tOF-t1,t3-t2,t2-tOF))

        return self._Rpos,self._tpos, self._scale[-1]

    def getScale(self,minFeatures=200):
        if len(self._newPoints) < minFeatures:
            self._nUpdateCount += 1
            updatePos = False
        else:
            updatePos = True
            alt = 1.5 + np.random.normal(0, 0.01)
            r_list = []
            for p1, p2 in zip(self._points, self._newPoints):
                x1, y1 = p1.squeeze()
                x2, y2 = p2.squeeze()
                FL_X = 2 * 0.8391 * alt
                FL_Y = 2 * 0.6249 * alt
                u = (FL_X / auv.IM_PRE_RESOLUTION[0]) * (y2 - y1)
                v = (FL_Y / auv.IM_PRE_RESOLUTION[1]) * (x2 - x1)

                r_list.append(math.sqrt(u ** 2 + v ** 2))

            self._scale.append(np.median(r_list))
        return updatePos

    def opticalFlowParams(self):

        return dict(winSize=(15, 15),
                    maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

    def update_motion(self,update=True):
        self._mask = None
        if update:
            try:
                E, self._VOmask = cv2.findEssentialMat(self._points, self._newPoints,
                                                       cameraMatrix=self._camMat,
                                                       method=cv2.RANSAC,prob = 0.999, #threshold=0.1,
                                                       mask=None)
            except:
                E = (0,0)
                self._VOmask = None
                pass
            if np.shape(E) == (3,3):
                retVal, R, t, _ = cv2.recoverPose(E, self._points, self._newPoints,
                                                  cameraMatrix=self._camMat,
                                                  mask=self._VOmask.copy())
            if retVal > 50:
                self._R = R.copy()
                self._t = t.copy()

        self.Rpos = np.dot(self._R,self._Rpos)
        self._tpos = self._tpos +  np.dot(self._R, self._tpos)
        self._tTrue += self._t
        self._C = getTransformationMat(self._R,self._t)



    def undistortPoints(self,distPoints):
        #distPoints = np.reshape(points,(-1,1,2))
        return cv2.undistortPoints(distPoints,distCoeffs=self._distMat,cameraMatrix=self._camMat)


