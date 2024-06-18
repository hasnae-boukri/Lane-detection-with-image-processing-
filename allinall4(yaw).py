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
    # Check if left_fit and right_fit are not None
    if left_fit is not None and right_fit is not None:
        # Generate y values for plotting based on the length of left_fit or right_fit
        ploty = np.linspace(0, 719, num=len(left_fit))

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

def calculate_yaw_angle(left_fit, right_fit):
    if left_fit is not None and right_fit is not None:
        # Calculate the angle of the lane lines
        left_angle = np.arctan(2 * left_fit[0] * 719 + left_fit[1])  # Assuming the image height is 720
        right_angle = np.arctan(2 * right_fit[0] * 719 + right_fit[1])

        # Calculate the average angle
        average_angle = (left_angle + right_angle) / 2

        # Calculate the yaw angle (angle between vehicle's longitudinal axis and lane direction)
        vehicle_yaw = 0  # Replace with your method to get the vehicle's yaw angle
        yaw_angle = average_angle - vehicle_yaw
        return np.degrees(yaw_angle)
    else:
        return None

def calculate_lateral_deviation(left_fit, right_fit):
    if left_fit is not None and right_fit is not None:
        # Calculate the x-coordinate of the lane center
        left_lane_bottom = left_fit[0] * 719 ** 2 + left_fit[1] * 719 + left_fit[2]
        right_lane_bottom = right_fit[0] * 719 ** 2 + right_fit[1] * 719 + right_fit[2]
        lane_center = (left_lane_bottom + right_lane_bottom) / 2

        # Calculate the lateral deviation
        image_center = 640 / 2  # Assuming the image width is 640
        lateral_deviation = lane_center - image_center
        
        # Convert lateral deviation from pixels to meters using calibration parameters
        lateral_deviation_meters = lateral_deviation * 3.7 / 700  # Adjust accordingly
        return lateral_deviation_meters
    else:
        return None

def undistort(img):
    #Load pickle
    dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

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

# Define previous fits
prev_left_fit = None
prev_right_fit = None

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

    # Find lane pixels using sliding windows
    leftx, lefty, rightx, righty = find_lane_pixels(mask)

    # Check if lane pixels are detected
    if leftx.any() and lefty.any():
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None

    if rightx.any() and righty.any():
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    # RANSAC algorithm for advanced lane line fitting
    if prev_left_fit is not None and prev_right_fit is not None:
        # Implement RANSAC fitting here
        pass  # Placeholder statement if no implementation yet

    # Update previous fits
    prev_left_fit = left_fit
    prev_right_fit = right_fit

    ## Draw the lane lines on the transformed frame (bird's eye view)
    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, undistorted_frame.shape[0]-1, undistorted_frame.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(undistorted_frame, np.int_([pts]), (0,255, 0))

    ## Calculate and display curvature
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    left_curverad, right_curverad = calculate_curvature(yvals, ym_per_pix, xm_per_pix, left_fit, right_fit)
    if left_curverad is not None and right_curverad is not None:
        cv2.putText(frame, 'Left Curvature: {:.2f} m'.format(left_curverad), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, 'Right Curvature: {:.2f} m'.format(right_curverad), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    
    ## Calculate and display yaw angle
    yaw_angle = calculate_yaw_angle(left_fit, right_fit)
    if yaw_angle is not None:
        cv2.putText(frame, 'Yaw Angle: {:.2f} degrees'.format(yaw_angle), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    ## Calculate and display lateral deviation
    lateral_deviation = calculate_lateral_deviation(left_fit, right_fit)
    if lateral_deviation is not None:
        cv2.putText(frame, 'Lateral Deviation: {:.2f} meters'.format(lateral_deviation), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Draw the bounding box
    cv2.line(frame, points[0], points[1], (0, 0, 255), 5)
    cv2.line(frame, points[1], points[2], (0, 0, 255), 5)
    cv2.line(frame, points[2], points[3], (0, 0, 255), 5)
    cv2.line(frame, points[3], points[0], (0, 0, 255), 5)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Undistorted Frame", undistorted_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
