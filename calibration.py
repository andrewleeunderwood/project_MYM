import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

CHECKERBOARD = (7,10)
CAMERA_FILE='camera.csv'
DIST_FILE='dis.csv'
print(CHECKERBOARD)
import numpy as np
import cv2

def rescale_frame(frame, percent=75):
		# width = int(frame.shape[1] * (percent / 100))
		# height = int(frame.shape[0] * (percent / 100))
		dim = (1000, 750)
		return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def do_main():
	cap = cv2.VideoCapture(0)
	print(cap)
	capture_num = 0

	while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()
			# print(frame)
			# Our operations on the frame come here
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			small_frame = rescale_frame(frame)

			# Display the resulting frame
			cv2.imshow('frame',  frame)
			if cv2.waitKey(1) & 0xFF == ord('s'):
					cv2.imwrite('calibration/frame' + str(capture_num) + '.jpeg', frame)
					print('Saved ' + str(capture_num))
					capture_num += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
	calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
	objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
	objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
	_img_shape = None
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	images = glob.glob('calibration/*.jpeg')
	print(images)
	for fname in images:
			img = cv2.imread(fname)
			if _img_shape == None:
					_img_shape = img.shape[:2]
			else:
					assert _img_shape == img.shape[:2], "All images must share the same size."
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			gray = cv2.convertScaleAbs(gray,alpha = 2)
			cv2.imwrite('gray.jpeg', gray)
			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
			print(ret,corners)
			# If found, add object points, image points (after refining them)
			if ret == True:
					objpoints.append(objp)
					cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
					imgpoints.append(corners)

	print('img end',objpoints)
	N_OK = len(objpoints)
	K = np.zeros((3, 3))
	D = np.zeros((4, 1))
	rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
	tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
	rms, k, d, r, t = cv2.calibrateCamera(
					objpoints,
					imgpoints,
					gray.shape[::-1],
					K,
					D,
					rvecs,
					tvecs,
					# calibration_flags,
					# (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
			)
	print("Found " + str(N_OK) + " valid images for calibration")
	print("DIM=" + str(_img_shape[::-1]))
	print("K=np.array(" + str(k.tolist()) + ")")
	print("D=np.array(" + str(d.tolist()) + ")")
	np.savetxt(CAMERA_FILE, k, delimiter =',',fmt="%0.14f")
	np.savetxt(DIST_FILE, d, delimiter =',',fmt="%0.14f")

def get_calibration_data():
	try:
		camera = np.loadtxt(CAMERA_FILE, delimiter=',')
		dist = np.loadtxt(DIST_FILE, delimiter=',')
	except Exception as e:
		raise e
	return camera, dist