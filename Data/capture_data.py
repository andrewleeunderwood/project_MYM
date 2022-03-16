import numpy as np
import cv2
import sys
import sys

from calibration import get_calibration_data
sys.path.append('../')
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
from calibration import get_calibration_data
def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(0)
print(cap)
capture_num = 24
frame_path='bordimg/'
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    camera,dist=get_calibration_data()
    frame = cv2.convertScaleAbs(frame,alpha = 1.1,beta=-30)
    frame = undistort(img=frame,DIM=(1920, 1080),K=np.array(camera),D=np.array(dist))
    pts = np.array( [
    [ [0,0], [0,1200], [50, 1200], [350,50] ,[1400,50],[1700,2000],[2000,2000],[2000,0]],
    # [ [0,0], [0,300], [1200, 300], [1200,0] ],
    ] )
    # dst = cv2.fillPoly(frame, pts =pts, color=(10,10,10))
    small_frame = rescale_frame(frame)

    # Display the resulting frame
    cv2.imshow('frame',  frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(frame_path+'frame' + str(capture_num) + '.jpeg', frame)
        print('Saved ' + str(capture_num))
        capture_num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()