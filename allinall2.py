import cv2
import numpy as np
import glob
import pickle

def calibrate_camera():
    nx = 9  # Number of inside corners in x
    ny = 6  # Number of inside corners in y

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (nx-1,ny-1,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Read calibration images
    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Chessboard not found in:", fname)

    # Calibrate the camera
    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    else:
        print("Unable to calibrate camera. Insufficient calibration images.")
        return None, None

def calculate_curvature(ploty, ym_per_pix, xm_per_pix, left_fit, right_fit):
    # Generate y values for plotting based on the length of left_fit or right_fit
    ploty = np.linspace(0, 719, num=len(left_fit))

    # Check if arrays are not empty
    if len(left_fit) > 0 and len(right_fit) > 0:
        # Interpolate the left and right fitted lines to match the length of ploty
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        # Calculate left and right lane curvature
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        
        # Define y-value where we want radius of curvature
        y_eval = np.max(ploty)
        
        # Calculate curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        
        return left_curverad, right_curverad
    else:
        print("Lane lines not detected. Curvature cannot be calculated.")
        return None, None

def undistort(img):
    #Load pickle
    dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted

# Camera Calibration
mtx, dist = calibrate_camera()

if mtx is None or dist is None:
    exit()

vidcap = cv2.VideoCapture("LaneVideo.mp4")
if not vidcap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Create trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, lambda x: None)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, lambda x: None)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: None)

yvals = None  # Define yvals globally

# Define the points of the bounding box
tl = (222,387)
bl = (70 ,472)
tr = (400,380)
br = (538,472)
points = [tl, tr, br, bl]

while True:
    success, image = vidcap.read()
    if not success:
        print("Error: Unable to read frame.")
        break
    
    frame = cv2.resize(image, (640,480))

    ## Choosing points for perspective transformation
    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    # Undistort frame
    undistorted_frame = undistort(transformed_frame)

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        y -= 40

    ## Define yvals
    yvals = np.arange(472, 0, -40)

    ## Draw the lane lines on the transformed frame (bird's eye view)
    if yvals is not None:
        for i in range(len(lx)-1):
            cv2.line(undistorted_frame, (lx[i], yvals[i]), (lx[i+1], yvals[i+1]), (0, 255, 0), 2)
        for i in range(len(rx)-1):
            cv2.line(undistorted_frame, (rx[i], yvals[i]), (rx[i+1], yvals[i+1]), (0, 255, 0), 2)

    ## Calculate and display curvature
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    left_curverad, right_curverad = calculate_curvature(yvals, ym_per_pix, xm_per_pix, np.array(lx), np.array(rx))
    if left_curverad is not None and right_curverad is not None:
        cv2.putText(undistorted_frame, 'Left Curvature: {:.2f} m'.format(left_curverad), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(undistorted_frame, 'Right Curvature: {:.2f} m'.format(right_curverad), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Draw the bounding box
    cv2.line(frame, points[0], points[1], (0, 0, 255), 5)
    cv2.line(frame, points[1], points[2], (0, 0, 255), 5)
    cv2.line(frame, points[2], points[3], (0, 0, 255), 5)
    cv2.line(frame, points[3], points[0], (0, 0, 255), 5)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Undistorted Frame", undistorted_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
