import cv2
def get_FAST_params():
    return dict(threshold = 80,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

def get_optical_flow_params():

    return dict(winSize = (15,15),
                           maxLevel = 4,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
def get_good_feature_params():
    return dict(maxCorners=2000,
                               qualityLevel=0.01,
                               minDistance=5,
                               blockSize=5,
                               useHarrisDetector=True)

def get_blob_params(): # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100;
    params.maxThreshold = 300;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    return params

