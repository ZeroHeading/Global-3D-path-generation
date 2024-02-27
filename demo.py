import os
from itertools import chain
from sklearn import linear_model
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale
import cv2 as cv
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from skimage import morphology
import cv2
import numpy as np


# field image path for global 3D path generation
filepath = r'.\Field_img'
# corresponding semantic segmentation map path
filepath2 = r'.\Pre_mask'

# COLORS
COLOR1 = (79, 199, 249)
COLOR2 = (0, 0, 255)
COLOR3 = (109, 190, 144)
COLOR4 = (71, 173, 112)
COLOR5 = (196, 114, 68)
COLOR6 = (0, 192, 255)
COLOR7 = (170, 170, 170)
COLOR8 = (49, 125, 237)
COLOR9 = (142, 209, 169)
COLOR10 = (230, 195, 157)
COLOR11 = (255, 255, 255)
COLOR12 = (0, 255, 255)
COLOR13 = (0, 192, 255)
COLOR14 = (240, 176, 0)


def translate(points, dx, dy):
    return [(int(x + dx), int(y + dy)) for (x, y) in points]


def rotate_point(point, angle, pivot):
    radian = math.radians(angle)
    offset_x = point[0] - pivot[0]
    offset_y = point[1] - pivot[1]
    rotated_x = offset_x * math.cos(radian) - offset_y * math.sin(radian)
    rotated_y = offset_x * math.sin(radian) + offset_y * math.cos(radian)
    x = round(rotated_x + pivot[0])
    y = round(rotated_y + pivot[1])
    return x, y


def rotate_points(points, angle, pivot):
    rotated_points = []
    for point in points:
        rotated_point = rotate_point(point, angle, pivot)
        rotated_points.append(rotated_point)
    return rotated_points


def list_reverse(L):
    pt_list = []
    L.reverse()
    for i in L:
        pt_list.append(i)
    return pt_list


def get_circle_pt(a, b, w, TR):
    m = (2 * math.pi) / w
    pt_0_90 = []
    pt_90_180 = []
    pt_180_270 = []
    pt_270_360 = []
    for i in range(0, round(w/4) + 1):
        x = round(a + TR * math.sin(m * i))
        y = round(b + TR * math.cos(m * i))
        pt_270_360.append([x, y])
    for i in range(round(w/4), round(w/4*2) + 1):
        x = round(a + TR * math.sin(m * i))
        y = round(b + TR * math.cos(m * i))
        pt_180_270.append([x, y])
    for i in range(round(w/4*2), round(w/4*3) + 1):
        x = round(a + TR * math.sin(m * i))
        y = round(b + TR * math.cos(m * i))
        pt_90_180.append([x, y])
    for i in range(round(w/4*3), round(w/4*4) + 1):
        x = round(a + TR * math.sin(m * i))
        y = round(b + TR * math.cos(m * i))
        pt_0_90.append([x, y])
    return pt_0_90, pt_90_180, pt_180_270, pt_270_360


def draw_start_turn_path(img, TR, C, T, pt_1, pt_2):
    pt_1_x, pt_1_y, pt_2_x, pt_2_y = pt_1[0], pt_1[1], pt_2[0], pt_2[1]
    circle_pt = []
    if pt_1_x > pt_2_x:
        L = pt_2_y - pt_1_y - TR*2
        cv.line(img, (pt_1_x, pt_1_y), (pt_2_x, pt_1_y), C, T, cv.LINE_AA)
        circle_pt.append([pt_1_x, pt_1_y])
        circle_pt.append([pt_2_x, pt_1_y])

        cv.ellipse(img, (pt_2_x, pt_1_y+TR), (TR, TR), 90, 90, 180, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_2_x, pt_1_y+TR, num_of_slice, TR)
        for i in l2:
            circle_pt.append(i)

        cv.line(img, (pt_2_x-TR, pt_1_y+TR), (pt_2_x-TR, pt_1_y+TR+L), C, T, cv.LINE_AA)
        circle_pt.append([pt_2_x-TR, pt_1_y+TR])
        circle_pt.append([pt_2_x-TR, pt_1_y+TR+L])

        cv.ellipse(img, (pt_2_x, pt_2_y-TR), (TR, TR), 90, 0, 90, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_2_x, pt_2_y-TR, num_of_slice, TR)
        for i in l1:
            circle_pt.append(i)

    if pt_1_x <= pt_2_x:
        L = pt_2_y - pt_1_y - TR*2
        cv.ellipse(img, (pt_1_x, pt_1_y+TR), (TR, TR), 90, 90, 180, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_1_x, pt_1_y+TR, num_of_slice, TR)
        for i in l2:
            circle_pt.append(i)

        cv.line(img, (pt_1_x-TR, pt_1_y+TR), (pt_1_x-TR, pt_1_y+TR+L), C, T, cv.LINE_AA)
        circle_pt.append([pt_2_x-TR, pt_1_y+TR])
        circle_pt.append([pt_2_x-TR, pt_1_y+TR+L])

        cv.ellipse(img, (pt_1_x, pt_2_y-TR), (TR, TR), 90, 0, 90, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_1_x, pt_2_y-TR, num_of_slice, TR)
        for i in l1:
            circle_pt.append(i)

        cv.line(img, (pt_1_x, pt_2_y), (pt_2_x, pt_2_y), C, T, cv.LINE_AA)
        circle_pt.append([pt_1_x, pt_2_y])
        circle_pt.append([pt_2_x, pt_2_y])
    return circle_pt


def draw_end_turn_path(img, TR, C, T, pt_1, pt_2):
    pt_1_x, pt_1_y, pt_2_x, pt_2_y = pt_1[0], pt_1[1], pt_2[0], pt_2[1]
    circle_pt = []
    if pt_1_x > pt_2_x:
        L = TR*2 - (pt_2_y - pt_1_y)

        cv.ellipse(img, (pt_1_x, pt_1_y+TR), (TR, TR), 90, 180, 270, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_1_x, pt_1_y+TR, num_of_slice, TR)
        l3.reverse()
        for i in l3:
            circle_pt.append(i)

        cv.line(img, (pt_1_x+TR, pt_1_y+TR), (pt_1_x+TR, pt_1_y+TR-L), C, T, cv.LINE_AA)
        circle_pt.append([pt_1_x+TR, pt_1_y+TR])
        circle_pt.append([pt_1_x+TR, pt_1_y+TR+L])

        cv.ellipse(img, (pt_1_x, pt_2_y-TR), (TR, TR), 90, 270, 360, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_1_x, pt_2_y-TR, num_of_slice, TR)
        l4.reverse()
        for i in l4:
            circle_pt.append(i)

        cv.line(img, (pt_2_x, pt_2_y), (pt_1_x, pt_2_y), C, T, cv.LINE_AA)
        circle_pt.append([pt_2_x, pt_2_y])
        circle_pt.append([pt_1_x, pt_2_y])

    if pt_1_x <= pt_2_x:
        L = pt_2_y - pt_1_y - TR*2
        cv.line(img, (pt_1_x, pt_1_y), (pt_2_x, pt_1_y), C, T, cv.LINE_AA)
        circle_pt.append([pt_1_x, pt_1_y])
        circle_pt.append([pt_2_x, pt_1_y])

        cv.ellipse(img, (pt_2_x, pt_1_y+TR), (TR, TR), 90, 180, 270, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_2_x, pt_1_y+TR, num_of_slice, TR)
        l3.reverse()
        for i in l3:
            circle_pt.append(i)

        cv.line(img, (pt_2_x+TR, pt_1_y+TR), (pt_2_x+TR, pt_1_y+TR+L), C, T, cv.LINE_AA)
        circle_pt.append([pt_2_x+TR, pt_1_y+TR])
        circle_pt.append([pt_2_x+TR, pt_1_y+TR+L])

        cv.ellipse(img, (pt_2_x, pt_2_y-TR), (TR, TR), 90, 270, 360, C, T, cv.LINE_AA)
        l1, l2, l3, l4 = get_circle_pt(pt_2_x, pt_2_y-TR, num_of_slice, TR)
        l4.reverse()
        for i in l4:
            circle_pt.append(i)
    return circle_pt


def draw_start_fishtail(img, TR, C1, C2, T, start_pt_list):
    for arc_num_1 in range(0, 1):
        circle_pt = []
        if start_pt_list[arc_num_1][0] < start_pt_list[arc_num_1+1][0]:
            cv.ellipse(img, (start_pt_list[arc_num_1][0], start_pt_list[arc_num_1][1]+TR), (TR, TR),
                       90, 90, 180, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(start_pt_list[arc_num_1][0], start_pt_list[arc_num_1][1]+TR, num_of_slice, TR)
            for i in l2:
                circle_pt.append(i)

            cv.line(img, (start_pt_list[arc_num_1][0]-TR, start_pt_list[arc_num_1+1][1]-TR),
                    (start_pt_list[arc_num_1][0]-TR, start_pt_list[arc_num_1][1]+TR), C1, T, cv.LINE_AA)
            circle_pt.append([start_pt_list[arc_num_1][0]-TR, start_pt_list[arc_num_1][1]+TR])
            circle_pt.append([start_pt_list[arc_num_1][0]-TR, start_pt_list[arc_num_1+1][1]-TR])

            cv.ellipse(img, (start_pt_list[arc_num_1][0], start_pt_list[arc_num_1+1][1]-TR), (TR, TR),
                       90, 0, 90, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(start_pt_list[arc_num_1][0], start_pt_list[arc_num_1+1][1]-TR, num_of_slice, TR)
            for i in l1:
                circle_pt.append(i)

            cv.line(img, (start_pt_list[arc_num_1][0], start_pt_list[arc_num_1+1][1]),
                    (start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1+1][1]), C1, T, cv.LINE_AA)
            circle_pt.append([start_pt_list[arc_num_1][0], start_pt_list[arc_num_1+1][1]])
            circle_pt.append([start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1+1][1]])

        if start_pt_list[arc_num_1][0] > start_pt_list[arc_num_1+1][0]:
            cv.line(img, (start_pt_list[arc_num_1][0], start_pt_list[arc_num_1][1]),
                    (start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1][1]), C1, T, cv.LINE_AA)
            circle_pt.append([start_pt_list[arc_num_1][0], start_pt_list[arc_num_1][1]])
            circle_pt.append([start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1][1]])

            cv.ellipse(img, (start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1][1]+TR), (TR, TR),
                       90, 90, 180, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1][1]+TR, num_of_slice, TR)
            for i in l2:
                circle_pt.append(i)

            cv.line(img, (start_pt_list[arc_num_1+1][0]-TR, start_pt_list[arc_num_1][1]+TR),
                    (start_pt_list[arc_num_1+1][0]-TR, start_pt_list[arc_num_1+1][1]-TR), C1, T, cv.LINE_AA)
            circle_pt.append([start_pt_list[arc_num_1+1][0]-TR, start_pt_list[arc_num_1][1]+TR])
            circle_pt.append([start_pt_list[arc_num_1+1][0]-TR, start_pt_list[arc_num_1+1][1]-TR])

            cv.ellipse(img, (start_pt_list[arc_num_1+1][0], start_pt_list[arc_num_1+1][1]-TR), (TR, TR),
                       90, 0, 90, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(start_pt_list[arc_num_1][0], start_pt_list[arc_num_1+1][1]-TR, num_of_slice, TR)
            for i in l1:
                circle_pt.append(i)
    return circle_pt


def draw_end_fishtail(img, TR, C1, C2, T, end_pt_list):
    for arc_num_2 in range(0, 1):
        circle_pt = []
        if end_pt_list[arc_num_2][0] < end_pt_list[arc_num_2 + 1][0]:
            cv.line(img, (end_pt_list[arc_num_2][0], end_pt_list[arc_num_2][1]),
                    (end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2][1]), C1, T, cv.LINE_AA)
            circle_pt.append([end_pt_list[arc_num_2][0], end_pt_list[arc_num_2][1]])
            circle_pt.append([end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2][1]])

            cv.ellipse(img, (end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2][1]+TR), (TR, TR),
                       90, 180, 270, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2][1]+TR, num_of_slice, TR)
            l3.reverse()
            for i in l3:
                circle_pt.append(i)
            cv.line(img, (end_pt_list[arc_num_2+1][0]+TR, end_pt_list[arc_num_2][1]+TR),
                    (end_pt_list[arc_num_2+1][0]+TR, end_pt_list[arc_num_2+1][1]-TR), C1, T, cv.LINE_AA)
            circle_pt.append([end_pt_list[arc_num_2+1][0]+TR, end_pt_list[arc_num_2][1]+TR])
            circle_pt.append([end_pt_list[arc_num_2+1][0]+TR, end_pt_list[arc_num_2+1][1]-TR])

            cv.ellipse(img, (end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2+1][1]-TR), (TR, TR),
                       90, 270, 360, C1, T, cv.LINE_AA)
            l1, l2, l3, l4 = get_circle_pt(end_pt_list[arc_num_2+1][0], end_pt_list[arc_num_2+1][1]-TR, num_of_slice, TR)
            l4.reverse()
            for i in l4:
                circle_pt.append(i)
    return circle_pt


def contours_anaylse(img, lower_limit):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < lower_limit:
            cv.drawContours(img, [cnt], 0, (0, 0, 0), -1)
    return img


def take_first(elem):
    return elem[0]


def linear_reg_with_points(X, Y, img_draw_line):
    list(chain.from_iterable(zip(Y, X)))
    plot_xy = list(chain(*zip(Y, X)))
    plot_xy = np.array(plot_xy).reshape(len(Y), 2).tolist()
    plot_xy.sort(key=take_first, reverse=True)
    plot_col_array = np.array([i[0] for i in plot_xy])
    plot_row_array = np.array([i[1] for i in plot_xy])
    plot_col_array_reshape = plot_col_array.reshape(-1, 1)
    model_linear = linear_model.LinearRegression()
    model_linear.fit(plot_col_array_reshape, plot_row_array)
    predict_row_list = model_linear.predict(plot_col_array_reshape)
    pt_1 = [predict_row_list[0], plot_col_array[0]]
    pt_2 = [predict_row_list[-1], plot_col_array[-1]]
    color = (np.random.randint(0, 255), np.random.randint(0, 255),
             np.random.randint(0, 255))
    if pt_1[0] != pt_2[0]:
        k = (pt_1[1] - pt_2[1]) / (pt_1[0] - pt_2[0])
        b = pt_1[1] - k * pt_1[0]
        p1 = (round((-b) / k), 0)
        p2 = (round((img_draw_line.shape[0] - b) / k), img_draw_line.shape[0])
        cv.line(img_draw_line, p1, p2, color, 2, cv.LINE_AA)
    if pt_1[0] == pt_2[0]:
        pt_1[0] += 0.001
        k = (pt_1[1] - pt_2[1]) / (pt_1[0] - pt_2[0])
        b = pt_1[1] - k * pt_1[0]
        p1 = (round((-b) / k), 0)
        p2 = (round((img_draw_line.shape[0] - b) / k), img_draw_line.shape[0])
        cv.line(img_draw_line, p1, p2, color, 2, cv.LINE_AA)
    return img_draw_line, pt_1, pt_2


def merge_list(L):
    L = list(map(set, L))
    lenth = len(L)
    for i in range(1, lenth):
        for j in range(i):
            if L[i] == {999999999} or L[j] == {999999999}:
                continue
            x = L[i].union(L[j])
            y = len(L[i]) + len(L[j])
            if len(x) < y:
                L[i] = x
                L[j] = {999999999}
    return [i for i in L if i != {999999999}]


filelist = []
list1 = os.listdir(filepath)
for file1 in list1:
    path = os.path.join(filepath, file1)
    filelist.append(path)

filelist2 = []
list2 = os.listdir(filepath2)
for file2 in list2:
    path2 = os.path.join(filepath2, file2)
    filelist2.append(path2)

for file, file_name, file2 in zip(filelist, list1, filelist2):
    mask_org = cv.imread(file)
    mask_org = cv.imdecode(np.fromfile(file, dtype=np.uint8), 1)

    color_img_org = cv.imread(file2)
    color_img_org = cv.imdecode(np.fromfile(file2, dtype=np.uint8), 1)

    RESIZE_v = 0.5
    org_color_img = color_img_org.copy()
    org_color_img = cv.resize(org_color_img, None, fx=RESIZE_v, fy=RESIZE_v)

    mask_org = cv.resize(mask_org, (1767, 1441))
    color_img_org = cv.resize(color_img_org, (1767, 1441))
    mask = np.zeros(((1441 + 400), (1767 + 400), 3), np.uint8)
    color_img = np.zeros(((1441 + 400), (1767 + 400), 3), np.uint8)

    mask[200:-200, 200:-200] = mask_org
    color_img[200:-200, 200:-200] = color_img_org

    mask = cv.resize(mask, None, fx=RESIZE_v, fy=RESIZE_v)
    color_img = cv.resize(color_img, None, fx=RESIZE_v, fy=RESIZE_v)

    org_color_img = color_img.copy()
    H, W, C = mask.shape

    black_pixels = np.where(
        (color_img[:, :, 0] == 0) &
        (color_img[:, :, 1] == 0) &
        (color_img[:, :, 2] == 0)
    )

    color_img[black_pixels] = [255, 255, 255]

    gray_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    thr_value, thr_mask = cv.threshold(gray_mask, 0, 255, cv.THRESH_OTSU)

    color_img[thr_mask>0] = (0,0,255)

    inv_bin = np.ones((H, W), np.uint8)*255
    inv_bin[thr_mask>0] = 0

    colored_mask = np.ones(mask.shape, np.uint8) * 255
    colored_mask[thr_mask>0] = (0,0,255)

    denoise_mask = thr_mask

    sinogram = radon(denoise_mask)
    plt.figure()
    plt.imshow(sinogram, cmap=plt.cm.Greys_r)

    _, y_corr_to_tilt = np.where(sinogram == np.max(sinogram))

    proj_curve = sinogram[:, y_corr_to_tilt[0]]
    plt.figure()
    plt.plot(proj_curve)
    plt.grid(linestyle='--', linewidth=0.5)

    peaks, _ = find_peaks(proj_curve)
    prominences = peak_prominences(proj_curve, peaks)[0]
    plt.plot(peaks, proj_curve[peaks], "x")
    plt.tick_params(labelsize=15)

    R = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotate_angle = np.argmax(R)

    M = cv.getRotationMatrix2D((W / 2, H / 2), 90 - rotate_angle, 1)

    warpped_img = cv.warpAffine(denoise_mask, M, (W, H))
    warpped_inv_bin = cv.warpAffine(inv_bin, M, (W, H))
    warpped_color = cv.warpAffine(color_img, M, (W, H), borderValue=(255,255,255))

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(warpped_img, connectivity=8)

    all_area = []
    for i in range(1, num_labels):
        out_img = np.zeros((H, W), np.uint8)
        cur_area = []
        mask = labels == i
        out_img[:, :][mask] = 255
        x_pos, y_pos = (out_img > 0).nonzero()
        for k in range(len(x_pos)):
            cur_area.append([y_pos[k], x_pos[k]])
        all_area.append(cur_area)

    warpped_color_for_save = warpped_color.copy()
    base_flag = 100000
    tole_v = 5
    connected_line_area = []
    for q in range(len(all_area)):
        color_1 = (np.random.randint(0, 255), np.random.randint(0, 255),
                 np.random.randint(0, 255))
        all_area[q] = np.array(all_area[q])
        MAX_Y = max(all_area[q][:, 1:])
        MIN_Y = min(all_area[q][:, 1:])
        for peak_1 in peaks:
            if MIN_Y[0] - tole_v <= (H - peak_1) <= MAX_Y[0] + tole_v:
                cv.line(warpped_color_for_save, (0, H - peak_1), (W, H - peak_1), color_1, 1, cv.LINE_AA)
                connected_line_area.append([q+1, base_flag+peak_1])

    before_merge = connected_line_area
    after_merge = merge_list(before_merge)
    new_connected_line_area = []
    for son_set in after_merge:
        trans_L = list(son_set)
        trans_L.sort()
        new_connected_line_area.append(trans_L)

    clustered_area = []
    for son_connect in new_connected_line_area:
        newnums = list(filter(lambda x: x < base_flag, son_connect))
        clustered_area.append(newnums)

    clustered_img = np.zeros((H, W, C), np.uint8)
    labels = np.array(labels, np.uint8)
    for area_idx in clustered_area:
        color_2 = (np.random.randint(0, 255), np.random.randint(0, 255),
                 np.random.randint(0, 255))
        for idx_num in area_idx:
            clustered_img[labels == idx_num] = color_2

    ske_bin = np.zeros((H, W), np.uint8)
    ske_bin[warpped_img>0] = 1

    skeleton = morphology.skeletonize(ske_bin)
    skeleton = skeleton.astype(np.uint8) * 255
    skeleton[skeleton > 0]=255
    warpped_color[skeleton>0]=(255,255,255)

    num_labels_skel, labels_skel, stats_skel, centroids_skel = cv.connectedComponentsWithStats(skeleton, connectivity=8)

    all_area_skel = []
    for ii in range(1, num_labels_skel):
        out_img_skel = np.zeros((H, W), np.uint8)
        cur_area_skel = []
        mask_skel = labels_skel == ii
        out_img_skel[:, :][mask_skel] = 255
        x_pos_skel, y_pos_skel = (out_img_skel > 0).nonzero()
        for kk in range(len(x_pos_skel)):
            cur_area_skel.append([y_pos_skel[kk], x_pos_skel[kk]])
        all_area_skel.append(cur_area_skel)

    for area_idx_skel in clustered_area:
        if len(area_idx_skel) > 1:
            temp_compare = []
            for f in area_idx_skel:
                temp_compare.append([f, centroids_skel[f][0]])
            temp_compare = sorted(temp_compare, key=lambda x: (x[1]))
            pt_2_joint = []
            for d in range(len(temp_compare)):
                if d == 0:
                    trans_all_area_skel = all_area_skel[temp_compare[d][0]-1]
                    sorted_all_point = sorted(trans_all_area_skel, key=lambda x: (x[0]))
                    pt_2_joint.append(sorted_all_point[-1])
                if d == (len(temp_compare)-1):
                    trans_all_area_skel = all_area_skel[temp_compare[d][0]-1]
                    sorted_all_point = sorted(trans_all_area_skel, key=lambda x: (x[0]))
                    pt_2_joint.append(sorted_all_point[0])
                if d != 0 and d != (len(temp_compare)-1):
                    trans_all_area_skel = all_area_skel[temp_compare[d][0]-1]
                    sorted_all_point = sorted(trans_all_area_skel, key=lambda x: (x[0]))
                    pt_2_joint.append(sorted_all_point[0])
                    pt_2_joint.append(sorted_all_point[-1])
            for pt_num in range(0, len(pt_2_joint), 2):
                cv.line(skeleton, tuple(pt_2_joint[pt_num]), tuple(pt_2_joint[pt_num+1]), 255, 1)
    colored_skeleton = np.ones((H, W, C), np.uint8)*255
    colored_skeleton[skeleton>0] = (213,155,91)

    bigger_H = round(H)
    bigger_W = round(W)
    plan_img = np.zeros((bigger_H, bigger_W), np.uint8)
    plan_img = skeleton

    num_labels_C, labels_C, stats_C, centroids_C = cv.connectedComponentsWithStats(plan_img, connectivity=8)

    all_area_C = []
    for cc in range(1, num_labels_C):
        out_img_C = np.zeros(plan_img.shape, np.uint8)
        cur_area_C = []
        mask_C = labels_C == cc
        out_img_C[:, :][mask_C] = 255
        x_pos_C, y_pos_C = (out_img_C > 0).nonzero()
        for uu in range(len(x_pos_C)):
            cur_area_C.append([y_pos_C[uu], x_pos_C[uu]])
        all_area_C.append(cur_area_C)

    white_img = np.ones((bigger_H, bigger_W, 3), np.uint8)
    one_list = []
    add_line_list = []
    start_pt_list = []
    end_pt_list = []
    num_flag = 0
    for a_line in all_area_C:
        a_line = np.array(a_line)
        sorted_a_line = sorted(a_line, key=lambda x: (x[0]))
        start_pt = sorted_a_line[0]
        end_pt = sorted_a_line[-1]

        extend_l = 0

        start_pt_list.append([start_pt[0]-extend_l, start_pt[1]])
        end_pt_list.append([end_pt[0]+extend_l, end_pt[1]])

        pure_centerline = sorted_a_line
        pure_centerline = np.array(pure_centerline)

        add_line_S = np.full((extend_l, 2), start_pt[1], dtype=int)
        add_line_S[:, 0] = np.arange((start_pt[0] - extend_l), start_pt[0])
        add_line_E = np.full((extend_l, 2), end_pt[1], dtype=int)
        add_line_E[:, 0] = np.arange(end_pt[0], (end_pt[0]+extend_l))

        sorted_a_line = np.vstack((add_line_S,sorted_a_line))
        sorted_a_line = np.vstack((sorted_a_line,add_line_E))
        a_x = sorted_a_line[:,0]
        a_y = sorted_a_line[:,1]
        plan_img[a_y, a_x] = 255
        add_line_list.append(sorted_a_line)

        one_list_son = sorted_a_line.tolist()
        one_list.append(one_list_son)

        center_x = pure_centerline[:,0]
        center_y = pure_centerline[:,1]

        if num_flag % 2 == 0:
            for dot_num_2 in range(len(center_x)):
                cv.circle(warpped_color, (center_x[dot_num_2], center_y[dot_num_2]),
                          1, COLOR2, -1, cv.LINE_AA)
        if num_flag % 2 != 0:
            for dot_num_2 in range(len(center_x)):
                cv.circle(warpped_color, (center_x[dot_num_2], center_y[dot_num_2]),
                          1, COLOR1, -1, cv.LINE_AA)
        num_flag += 1

    TR = 20
    line_width = 1
    num_of_slice = 500
    one_line = []
    for i in range(0, 13, 4):
        one_line.extend(one_list[i])

        circle_pt_1 = draw_end_turn_path(warpped_color, TR, COLOR4, line_width, end_pt_list[i], end_pt_list[i+2])
        one_line.extend(circle_pt_1)

        temp_l = list_reverse(one_list[i + 2])
        one_line.extend(temp_l)

        if i<12:
            circle_pt_2 = draw_start_turn_path(warpped_color, TR, COLOR4, line_width, start_pt_list[i+2], start_pt_list[i+4])
            one_line.extend(circle_pt_2)

    circle_pt_3 = draw_start_fishtail(warpped_color, TR, COLOR4, COLOR4, line_width, [start_pt_list[-2], start_pt_list[-1]])
    one_line.extend(circle_pt_3)
    #
    for i in range(0, 13, 4):
        one_line.extend(one_list[15-i])

        circle_pt_4 = draw_end_turn_path(warpped_color, TR, COLOR4, line_width, end_pt_list[15-i-2], end_pt_list[15-i])
        temp_l = list_reverse(circle_pt_4)
        one_line.extend(temp_l)

        temp_l = list_reverse(one_list[15-i-2])
        one_line.extend(temp_l)

        if i<12:
            circle_pt_5 = draw_start_turn_path(warpped_color, TR, COLOR4, line_width, start_pt_list[15-i-4], start_pt_list[15-i-2])
            temp_l = list_reverse(circle_pt_5)
            one_line.extend(temp_l)

    angle = 90 - rotate_angle
    pivot = [round(W/2), round(H/2)]
    rotated_points = rotate_points(one_line, angle, pivot)
    shift_rotated_points = translate(rotated_points, 200, 200)

    for i in range(0, len(rotated_points), 5):
        cv.circle(org_color_img, (rotated_points[i][0], rotated_points[i][1]), 1, 255, -1)
        cv.imshow('test_line', org_color_img)
        cv.waitKey(1)
    new_list = shift_rotated_points[::25]

    from osgeo import gdal
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    def get_data(file_path, col_pos, row_pos):
        dataset = gdal.Open(file_path)
        dem_XSize = dataset.RasterXSize
        dem_YSize = dataset.RasterYSize
        dem_bands = dataset.RasterCount
        band = dataset.GetRasterBand(1)
        dem_data = band.ReadAsArray(0, 0, dem_XSize, dem_YSize)
        dem_data_index = np.where(dem_data < 0)
        dem_data[dem_data_index] = None
        ele_pos_img = np.zeros((dem_YSize, dem_XSize), np.uint8)
        ele_pos_img[row_pos, col_pos] = 255
        ele_path = np.full([dem_YSize, dem_XSize], np.nan)
        ele_path[row_pos, col_pos] = dem_data[row_pos, col_pos]
        Z_value = dem_data[row_pos, col_pos]
        return dem_data, ele_path, dem_YSize, dem_XSize, Z_value

    elevation_map_path = r'D:\3D_detection\gray1.tif'

    col_pos = [point[0] for point in new_list]
    row_pos = [point[1] for point in new_list]
    dem_data, ele_path, dem_YSize, dem_XSize, Z_value = get_data(elevation_map_path, col_pos, row_pos)

    window_size = 5
    poly_degree = 2
    Z_value = savgol_filter(Z_value, window_size, poly_degree)

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(row_pos, col_pos, Z_value, 'gray')
    cm = plt.cm.get_cmap('viridis')
    ax1.scatter3D(row_pos, col_pos, Z_value, c=Z_value, cmap=cm)
    project_z = [90] * len(Z_value)
    ax1.plot(row_pos, col_pos, project_z)
    ax1.view_init(elev=35, azim=-125)
    plt.show()

    warpped_inv_bin = cv.cvtColor(warpped_inv_bin, cv.COLOR_GRAY2BGR)
    for peak in peaks:
        color_3 = (np.random.randint(0, 255), np.random.randint(0, 255),
                 np.random.randint(0, 255))
        cv.line(warpped_inv_bin, (0, H - peak), (W, H - peak), color_3, 1, cv.LINE_AA)

    cv.waitKey()
    cv.destroyAllWindows()
