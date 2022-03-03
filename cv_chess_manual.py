import re
import cv2
from keras.models import load_model
import sys
sys.path.append("./")
sys.path.append("./Data")
from cv_chess_functions import (read_img,
															 canny_edge,
															 hough_line,
															 h_v_lines,
															 line_intersections,
															 cluster_points,
															 augment_points,
															 write_crop_images,
															 grab_cell_files,
															 classify_cells,
															 fen_to_image,
															 atoi,undistort)
import numpy as np
from calibration import get_calibration_data
pt=[]
n=0
#マウスの操作があるとき呼ばれる関数
def mouse_callback(event, x, y, flags, param):
    global pt

    #マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        print('click!')
        print(x, y)
        print(pt)
        pt.append((x, y))

    #マウスの右ボタンがクリックされたとき
    if event == cv2.EVENT_RBUTTONDOWN:
        print(pt)
        pt.pop()
# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
		# width = int(frame.shape[1] * (percent / 100))
		# height = int(frame.shape[0] * (percent / 100))
		dim = (1000, 750)
		return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Find the number(s) in the text
def natural_keys(text):
		return [atoi(c) for c in re.split('(\d+)', text)]

print('read h5')
# Load in the CNN model
model = load_model('model_VGG16_weight.h5')
print('end h5')
# Select the live video stream source (0-webcam & 1-GoPro)
cap = cv2.VideoCapture(0)
cv2.namedWindow('live', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('live', mouse_callback)
# Show the starting board either as blank or with the initial setup
# start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
blank = '8/8/8/8/8/8/8/8'
board = fen_to_image(blank)
board_image = cv2.imread('current_board.png')
cv2.imshow('img', board_image)
print('endfen')
while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		# frame = undistort(frame)
		camera,dist=get_calibration_data()
		frame = cv2.convertScaleAbs(frame,alpha = 1.1,beta=-30)
		frame = undistort(img=frame,DIM=(1920, 1080),K=np.array(camera),D=np.array(dist))
		pts = np.array( [
[ [0,0], [0,1200], [50, 1200], [350,50] ,[1400,50],[1700,2000],[2000,2000],[2000,0]],
# [ [0,0], [0,300], [1200, 300], [1200,0] ],
] )
		dst = cv2.fillPoly(frame, pts =pts, color=(10,10,10))
		ptl = np.array( pt )

		# frame = dst
		# Resizes each frame
		small_frame = rescale_frame(frame)
		cv2.polylines(small_frame, [ptl] ,False,(200,10,10))
		# Display the resulting frame
		cv2.imshow('live', small_frame)

		if cv2.waitKey(1) & 0xFF == ord(' '):

				print('Working...')
				# Save the frame to be analyzed
				cv2.imwrite('frame.jpeg', frame)

				# Low-level CV techniques (grayscale & blur)
				img, gray_blur = read_img('frame.jpeg')
				# Canny algorithm
				edges = canny_edge(gray_blur)
				cv2.imwrite('edges.jpeg', edges)
				# Hough Transform
				lines = hough_line(edges)
				frame2 = frame.copy()

				if lines is None:
					print("line none")
					continue
				# Separate the lines into vertical and horizontal lines
				h_lines, v_lines = h_v_lines(lines)
				# Find and cluster the intersecting
				intersection_points = line_intersections(h_lines, v_lines)
				for _point in intersection_points:
					point = (int(_point[0]),int(_point[1]))
					cv2.drawMarker(frame2, position=point, color=(0, 255, 0))
				points = cluster_points(intersection_points)
				for _point in points:
					point = (int(_point[0]),int(_point[1]))
					cv2.drawMarker(frame2, position=point, color=(0, 0, 255))
				# Final coordinates of the board
				points = augment_points(points)
				print(points)
				for _point in points:
					point = (int(_point[0]),int(_point[1]))
					cv2.drawMarker(frame2, position=point, color=(255, 0, 0))
				cv2.imwrite('frame2.jpeg', frame2)
				# Crop the squares of the board a organize into a sorted list
				x_list = write_crop_images(img, points, 0)
				img_filename_list = grab_cell_files()
				print(img_filename_list)
				img_filename_list.sort(key=natural_keys)
				# Classify each square and output the board in Forsyth-Edwards Notation (FEN)
				fen = classify_cells(model, img_filename_list)
				# Create and save the board image from the FEN
				board = fen_to_image(fen)
				# Display the board in ASCII
				print(board)
				# Display and save the board image
				board_image = cv2.imread('current_board.png')
				cv2.imshow('img', board_image)
				print('Completed!')

		if cv2.waitKey(1) & 0xFF == ord('q'):
				# End the program
				break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
