import glob
# import re
import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean


# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file), 1)

    W = 1000
    height, width, depth = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

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


# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
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


# Average the y value in each row and augment original point
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


# Crop board into separate images
def write_crop_images(img, points, img_count, folder_path='./raw_data/'):
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
            cv2.imwrite('./raw_data/alpha_data_image' + str(img_count) + '.jpeg', cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_count


# Create a list of image file names
img_filename_list = []
folder_name = './test_data/*'
for path_name in glob.glob(folder_name):
    # file_name = re.search("[\w-]+\.\w+", path_name) (use if in same folder)
    img_filename_list.append(path_name)  # file_name.group()

# Create and save cropped images from original images to the data folder
img_count = 20000
print_number = 0
for file_name in img_filename_list:
    print(file_name)
    img, gray_blur = read_img(file_name)
    print(np.shape(img))
    print(np.shape(gray_blur))
    edges = canny_edge(gray_blur)
    print('edges: ' + str(np.shape(edges)))
    lines = hough_line(edges)
    print('line: ' + str(np.shape(lines)))
    h_lines, v_lines = h_v_lines(lines)
    assert len(h_lines) >= 11
    assert len(v_lines) >= 11
    print('h_lines: ' + str(np.shape(h_lines)))
    print('v_lines: ' + str(np.shape(v_lines)))
    intersection_points = line_intersections(h_lines, v_lines)
    print('lines: ' + str(np.shape(intersection_points)))
    points = cluster_points(intersection_points)
    # if np.shape(points)[0] < 100:
    #     continue
    points = augment_points(points)
    print('points: ' + str(np.shape(points)))
    img_count = write_crop_images(img, points, img_count)
    print('img_count: ' + str(img_count))
    print('PRINTED')
    print_number += 1
print(print_number)
