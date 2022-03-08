from configparser import ExtendedInterpolation
import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import re
import glob
import PIL
def linear_func(x1,y1,x2,y2):
    a = (y2-y1)/(x2-x1)
    b=y1-a*x1
    return {'a':a,'b':b}
# 線分ABと線分CDの交点を求める関数
def _calc_cross_point(pointA, pointB, pointC, pointD):
    cross_point = (0,0)
    bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

    # 直線が平行な場合
    if (bunbo == 0):
        return False, cross_point

    vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
    r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
    s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo

    # rを使った計算の場合
    distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
    cross_point = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))

    # sを使った計算の場合
    # distance = ((pointD[0] - pointC[0]) * s, (pointD[1] - pointC[1]) * s)
    # cross_point = (int(pointC[0] + distance[0]), int(pointC[1] + distance[1]))

    return True, cross_point
# points=(p1,p2,p3,p4)
# ４の点からチェス盤のマス目のポイントを得る関数
def points_by_points(points,X=9):
    print(points)
    if points[2][0]>points[3][0]:
        points[2],points[3]=points[3],points[2]
    # t_l=(math.fabs(points[0][0]-points[1][0])**2+math.fabs(points[0][1]-points[1][1])**2)**0.5
    # b_l=(math.fabs(points[2][0]-points[3][0])**2+math.fabs(points[2][1]-points[3][1])**2)**0.5
    # height = (math.fabs(points[0][0]-points[2][0])**2+math.fabs(points[1][1]-points[3][1])**2)**0.5
    # print('t_l',t_l)
    # print('b_l',b_l)
    # print('height',height)
    # h_a=t_l+b_l/2
    # s=(t_l/b_l)/height
    #print('s',s)
    h_t_lines = np.column_stack([np.linspace(points[0][0], points[1][0], X),np.linspace(points[0][1], points[1][1], X)])
    h_b_lines = np.column_stack([np.linspace(points[2][0], points[3][0], X),np.linspace(points[2][1], points[3][1], X)])
    h_b_lines=np.sort(h_b_lines,axis=0)
    retpoints=[]
    b=math.fabs(points[0][0]-points[1][0])/math.fabs(points[2][0]-points[3][0])
    for i in range(0,len(h_t_lines)):
        print(h_t_lines[i][0], h_b_lines[i][0])
        _points = np.column_stack([np.linspace(h_t_lines[i][0], h_b_lines[i][0], X),np.linspace(h_t_lines[i][1], h_b_lines[i][1], X)])
        retpoints.append(_points)
    print(retpoints)
    return retpoints

# 角ABCを計算(rad)
def get_angle(_pointA, _pointB, _pointC):
    pointA, pointB, pointC = np.array(_pointA,dtype=float), np.array(_pointB,dtype=float), np.array(_pointC,dtype=float)
    vec_BA = pointA - pointB
    vec_BC = pointC - pointB
    cos_ABC = np.inner(vec_BA, vec_BC) / (np.linalg.norm(vec_BA) * np.linalg.norm(vec_BC))
    return np.arccos(cos_ABC)

# AをOを中心に (反時計回りに)angle度回転
def rotate(_pointA, _pointO, angle):
    pointA, pointO = np.array(_pointA,dtype=float), np.array(_pointO,dtype=float)
    pointA -= pointO
    pointA = np.array([pointA[0] * np.cos(angle) - pointA[1] * np.sin(angle), pointA[0] * np.sin(angle) + pointA[1] * np.cos(angle)])
    pointA += pointO
    return pointA

# 一辺がA-M-B, もう一辺がC-D と並んでいることを仮定 
def partition_by_ratio(_pointA, _pointB, _pointC, _pointD, _pointM, X=9):
    pointA, pointB, pointC, pointD, pointM = np.array(_pointA,dtype=float), np.array(_pointB,dtype=float), np.array(_pointC,dtype=float), np.array(_pointD,dtype=float), np.array(_pointM,dtype=float)
    #infinity_point = np.array(_calc_cross_point(pointA, pointB, pointC, pointD)[1],dtype=float)

    ratio = np.power(np.linalg.norm(pointB - pointM) / np.linalg.norm(pointM - pointA), 0.25)
    initial = 1.0 * (1.0 - ratio) / (1.0 - np.power(ratio, X - 1))
    print(pointA,pointB,pointC,pointD,pointM,pointB-pointM,pointM-pointA)
    print(initial, ratio)
    ret_lines = np.zeros([9,2,2],dtype=float)
    vec_sum = 0
    vec = (pointB - pointA) * initial
    for i in range(X):
        ret_lines[i][0] = pointA + vec_sum
        vec_sum += vec
        vec *= ratio

    vec_sum = 0
    vec = (pointD - pointC) * initial

    for i in range(X):
        ret_lines[i][1] = pointC + vec_sum
        vec_sum += vec
        vec *= ratio

    print(ret_lines)
    return ret_lines

# points=(p1,p2,p3,p4)
# ４の点からチェス盤のマス目のポイントを得る関数(二点透視図法による)
# points[0],points[1],points[2]が一列にならんでおり、points[0],points[3],points[4]が一列にならんでおり、points[5]が残りの角である
def points_by_points_with_twopoint_perspective(points,X=9):
    print(points)
    if points[4][0]>points[5][0]:
        points[4],points[5]=points[5],points[4]
    
    horizontal_lines=partition_by_ratio(points[0],points[2],points[4],points[5],points[1],X)
    vertical_lines=partition_by_ratio(points[0],points[4],points[2],points[5],points[3],X)
    ret_points = np.zeros([9,9,2])
    for i in range(X):
        for j in range(X):
            ret_points[i][j] = np.array(_calc_cross_point(horizontal_lines[i][0], horizontal_lines[i][1], vertical_lines[j][0], vertical_lines[j][1])[1])
    print(ret_points)
    return ret_points

# ４の点からチェス盤のマス目のポイントを得る関数(透視投影行列による)
def points_by_points_with_perspective_projection(points,P=9):
    print(points)
    if points[2][0]>points[3][0]:
        points[2],points[3]=points[3],points[2]
    matrix_size = 12
    coefficient_matrix = np.zeros([matrix_size,matrix_size],dtype=float)
    extended_matrix = np.zeros([matrix_size,1],dtype=float)
    for i in range(2):
        for j in range(2):
            index = i * 2 + j
            x = points[index][0]
            y = points[index][1]
            X = i * 9.0
            Y = j * 9.0
            Z = 1.0
            print(i,j,index,x,y,X,Y,Z)
            coefficient_matrix[index * 3][0] = X
            coefficient_matrix[index * 3][1] = Y
            coefficient_matrix[index * 3][2] = Z
            coefficient_matrix[index * 3][3] = 1
            extended_matrix[index * 3] = x
            coefficient_matrix[index * 3 + 1][4] = X
            coefficient_matrix[index * 3 + 1][5] = Y
            coefficient_matrix[index * 3 + 1][6] = Z
            coefficient_matrix[index * 3 + 1][7] = 1
            extended_matrix[index * 3 + 1] = y
            coefficient_matrix[index * 3 + 2][8] = X
            coefficient_matrix[index * 3 + 2][9] = Y
            coefficient_matrix[index * 3 + 2][10] = Z
            coefficient_matrix[index * 3 + 2][11] = 1
            extended_matrix[index * 3 + 2] = 1
    for i in range(12):
        coefficient_matrix[i][i] += 0.000000001
    print(coefficient_matrix)
    print(extended_matrix)
    answer = np.linalg.solve(coefficient_matrix, extended_matrix)
    print(answer)
    print(answer.shape)
    perspective_projection_matrix = np.reshape(answer,[3, 4])
    print(perspective_projection_matrix)
    ret_points = []
    for i in range(P):
        for j in range(P):
            X = i * 9.0
            Y = j * 9.0
            Z = 1.0
            point = perspective_projection_matrix * np.matrix([[X], [Y], [Z], [1]])
            print(point)
            ret_points.append(np.array([float(point[0][0]), float(point[1][0])], dtype=float))
            print(float(point[0][0]))
    print(ret_points)
    return ret_points

# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    return img, gray_blur


# Canny edge detection
def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges
    # return cv2.Canny(img,50,150,apertureSize = 3)


# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    if lines is None:
      return None
    lines = np.reshape(lines, (-1, 2))
    return lines


# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


# Average the y value in each row and augment original points
def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


# Crop board into separate images and write to folder
def write_crop_images(img, points, img_count=0, folder_path='./Data/raw_data/'):
    num_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            # ratio_h = 2
            # ratio_w = 1
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_count += 1
            cv2.imwrite(folder_path + str(img_count) + '.jpeg', cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_count


# Crop board into separate images and shows
def x_crop_images(img, points):
    num_list = []
    img_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_list.append(cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_list


# Convert image from RGB to BGR
def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data


# Adjust image into (1, 224, 224, 3)
def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im


# Changes digits in text to ints
def atoi(text):
    return int(text) if text.isdigit() else text


# Finds the digits in a string
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Reads in the cropped images to a list
def grab_cell_files(folder_name='./Data/raw_data/*'):
    img_filename_list = []
    for path_name in glob.glob(folder_name):
        img_filename_list.append(path_name)
    # img_filename_list = img_filename_list.sort(key=natural_keys)
    return img_filename_list


# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells(model, img_filename_list):
    category_reference = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: '1', 7: 'B', 8: 'K', 9: 'N', 10: 'P',
                          11: 'Q', 12: 'R'}
    pred_list = []
    for filename in img_filename_list:
        img = prepare_image(filename)
        out = model.predict(img)
        top_pred = np.argmax(out)
        pred = category_reference[top_pred]
        pred_list.append(pred)

    fen = ''.join(pred_list)
    fen = fen[::-1]
    fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    sum_digits = 0
    for i, p in enumerate(fen):
        if p.isdigit():
            sum_digits += 1
        elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
            fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
            sum_digits = 0
    if sum_digits > 1:
        fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    fen = fen.replace('D', '')
    return fen


# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")
    return board
def undistort(img=None,DIM=(1920, 1080),K=np.array([[  1.08515378e+03,   0.00000000e+00,   9.83885166e+02],
       [  0.00000000e+00,   1.08781630e+03,   5.36422266e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]),D=np.array([[0.0], [0.0], [0.0], [0.0]])):
    # h,w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)