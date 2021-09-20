import cv2 as cv
import numpy as np
import pandas as pd
from copy import copy
import figure_factory as ff
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.signal import find_peaks


def crop_number(im):
    _, im_th = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    wh = np.where(im_th == 255)
    y_min = np.min(wh[0])
    y_max = np.max(wh[0])
    x_min = np.min(wh[1])
    x_max = np.max(wh[1])
    im_crop = im_th[y_min:y_max, x_min:x_max]

    return im_crop


def get_projections(im_crop):
    x_proj = np.sum(im_crop, axis=0)
    x_proj = x_proj / np.linalg.norm(x_proj)

    y_proj = np.sum(im_crop, axis=1)
    y_proj = y_proj / np.linalg.norm(y_proj)

    return x_proj, y_proj


def get_refs():
    refs = {}
    for i in range(1, 10):
        im_file = f'ref/ref_{i}.png'

        im = cv.imread(im_file)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        im_crop = crop_number(im)

        refs[i] = get_projections(im_crop)

    return refs


def interpolate(x, x_ref):
    x_mod = interp.interp1d(np.arange(x.size), x)
    return x_mod(np.linspace(0, x.size - 1, x_ref.size))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def get_rotated_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Binarizing
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, im_bin = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    im_bin = ~im_bin

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    opening = cv.morphologyEx(im_bin, cv.MORPH_OPEN, kernel)

    output = cv.connectedComponentsWithStats(opening, 4, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    argmax = np.argsort(stats[:, cv.CC_STAT_AREA])[-2]

    component_mask = (labels == argmax).astype("uint8") * 255

    # find outer contour
    cntrs = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # get rotated rectangle from outer contour
    rotrect = cv.minAreaRect(cntrs[0])
    box = cv.boxPoints(rotrect)
    box = np.int0(box)

    # draw rotated rectangle on copy of img as result
    result = img.copy()
    cv.drawContours(result, [box], 0, (0, 0, 255), 2)

    # get angle from rotated rectangle
    angle = rotrect[-1]

    # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    print(angle, "deg")
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if angle > 45:
        angle = angle % 90

    if angle < -45:
        angle = angle % 90
    print(angle, "deg")

    return rotate_image(img, -angle)


def get_transformed_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Binarizing
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, im_bin = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    im_bin = ~im_bin

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))

    # dilate the image to get text
    opening = cv.morphologyEx(im_bin, cv.MORPH_OPEN, kernel)

    output = cv.connectedComponentsWithStats(opening, 4, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    argmax = np.argsort(stats[:, cv.CC_STAT_AREA])[-2]
    component_mask = (labels == argmax).astype("uint8") * 255

    top_left = []
    min_val = 1000000000
    for i in range(component_mask.shape[0]):
        if i * i > min_val:
            break
        for j in range(component_mask.shape[1]):
            if j * j > min_val:
                break
            if component_mask[i, j] > 0:
                val = i * i + j * j
                if val < min_val:
                    min_val = val
                    top_left = [j - 20, i - 20]

    bottom_left = []
    min_val = 1000000000
    for i in reversed(range(component_mask.shape[0])):
        if (height - i) * (height - i) > min_val:
            continue
        for j in range(component_mask.shape[1]):
            if j * j > min_val:
                break
            if component_mask[i, j] > 0:
                val = (height - i) * (height - i) + j * j
                if val < min_val:
                    min_val = val
                    bottom_left = [j - 20, i + 20]

    bottom_right = []
    min_val = 1000000000
    for i in reversed(range(component_mask.shape[0])):
        if (height - i) * (height - i) > min_val:
            break
        for j in reversed(range(component_mask.shape[1])):
            if (width - j) * (width - j) > min_val:
                break
            if component_mask[i, j] > 0:
                val = (height - i) * (height - i) + (width - j) * (width - j)
                if val < min_val:
                    min_val = val
                    bottom_right = [j + 20, i + 20]

    top_right = []
    min_val = 1000000000
    for i in range(component_mask.shape[0]):
        if i * i > min_val:
            break
        for j in reversed(range(component_mask.shape[1])):
            if (width - j) * (width - j) > min_val:
                break
            if component_mask[i, j] > 0:
                val = i * i + (width - j) * (width - j)
                if val < min_val:
                    min_val = val
                    top_right = [j + 20, i - 20]

    print('bottom_left', bottom_left)
    print('bottom_right', bottom_right)
    print('top_left', top_left)
    print('top_right', top_right)

    # pts1 = np.float32([[35,170],[824,44],[47,933],[864,918]])
    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 80], [800, 0], [0, 800], [800, 800]])

    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (800, 800))

    return dst


class SudokuImage:
    def __init__(self, im_file):
        self.im_file = im_file
        self.im_orig = cv.imread(im_file)
        # self.im = get_rotated_image(self.im_orig)
        self.im = get_transformed_image(self.im_orig)
        self.im_gray = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)
        self.height, self.width = self.im_gray.shape

        self.refs = get_refs()

        self.sudoku_df = None  # will contain the sudoku grid
        self.im_bin = None  # will contain the binarized image

        self.v_lines = None  # will contain the vertical lines of the grid
        self.h_lines = None  # will contain the horizontal lines of the grid

    def get_rotated_image_figure(self):
        return ff.get_rotated_image_figure(self)

    def get_grid_lines_figure(self):
        return ff.get_grid_lines_figure(self)

    def get_numbers_detection(self):
        return ff.get_numbers_detection(self)

    def extract_sudoku(self):
        self.binarize()
        # self.detect_grid()
        self.detect_grid_v2()
        self.detect_numbers()

        return self.sudoku_df

    def binarize(self):
        # blur = cv.GaussianBlur(self.im_gray, (5, 5), 0)  # todo: is this mandatory?
        _, im_bin = cv.threshold(self.im_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        self.im_bin = ~im_bin

    def detect_grid_v2(self):
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # reduced size after replacing rotating step
        opening = cv.morphologyEx(self.im_bin, cv.MORPH_OPEN, kernel)

        # Finding connected components
        output = cv.connectedComponentsWithStats(opening, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # Taking the second one in area (the border of the grid
        argmax = np.argsort(stats[:, cv.CC_STAT_AREA])[-2]

        # Extracting its mask
        componentMask = (labels == argmax).astype("uint8") * 255

        # Filling it
        im_floodfill = componentMask.copy()
        h, w = componentMask.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv.bitwise_not(im_floodfill)

        im_out = componentMask | im_floodfill_inv  # we could try to extract the average step between cells from here ? depending on the rectangle shape

        # im_grid_only = im_bin[im_out == 255]
        self.im_bin[im_out == 0] = 0  # Keeping only the mask

        x_coords = []
        evo_x = []
        for i in range(self.im_bin.shape[0]):
            evo = self.im_bin[i, :]
            evo_diff = np.diff(evo)
            if len(evo_diff[evo_diff == 255]) == 10:
                if len(evo_x) == 0:
                    evo_x = evo.astype(float)
                else:
                    evo_x += evo.astype(float)
                x_coords.append(np.where(evo_diff == 255)[0].tolist())

        # plt.figure(1)
        # for x in x_coords:
        #     plt.scatter(list(range(len(x))), x)
        # # plt.show()

        y_coords = []
        evo_y = []
        for i in range(self.im_bin.shape[1]):
            evo = self.im_bin[:, i]
            evo_diff = np.diff(evo)
            if len(evo_diff[evo_diff == 255]) == 10:
                if len(evo_y) == 0:
                    evo_y = evo.astype(float)
                else:
                    evo_y += evo.astype(float)
                y_coords.append(np.where(evo_diff == 255)[0].tolist())

        # plt.figure(2)
        # for y in y_coords:
        #     plt.scatter(list(range(len(y))), y)
        # # plt.show()
        # print(x_coords)
        # print(y_coords)

        px = find_peaks(evo_x, distance=60)[0]
        py = find_peaks(evo_y, distance=60)[0]

        # plt.figure()
        # plt.plot(evo_x, color='k')
        #
        # for ppx in px:
        #     plt.plot([ppx, ppx], [0, np.max(evo_x)])
        #
        # plt.figure()
        # plt.plot(evo_y, color='k')
        # for ppy in py:
        #     plt.plot([ppy, ppy], [0, np.max(evo_y)])
        #
        # plt.show()

        # x_coords = np.median(x_coords, axis=0) + 1  # coming from the diff
        # y_coords = np.median(y_coords, axis=0) + 1  # coming from the diff

        x_coords = px[-10:]
        y_coords = py[-10:]

        # plt.figure(1)
        # plt.plot(list(range(len(x_coords))), x_coords, color='r')
        #
        # plt.figure(2)
        # plt.plot(list(range(len(y_coords))), y_coords, color='r')
        # plt.show()
        #
        # print(x_coords)
        # print(y_coords)

        self.v_lines = [(int(x), int(x), 0, self.height - 1) for x in x_coords]
        self.h_lines = [(0, self.width - 1, int(y), int(y)) for y in y_coords]

    def detect_grid(self):
        if self.im_bin is None:
            raise AttributeError('No binarized image found')

        # Corners + lines detection
        edges = cv.Canny(self.im_gray, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

        tol_x = int(self.width / 5)
        tol_y = int(self.height / 5)

        x_top = []
        x_bot = []
        y_left = []
        y_right = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            if np.abs(x1 - x2) > tol_x and np.abs(y1 - y2) > tol_y:
                print(np.abs(x1 - x2), tol_x, np.abs(y1 - y2), tol_y)
                pass
            else:
                x_top.append(x1)
                x_bot.append(x2)
                y_left.append(y1)
                y_right.append(y2)

        v_lines = []
        h_lines = []
        for xt, xb, yl, yr in zip(x_top, x_bot, y_left, y_right):
            if abs(yl - yr) > abs(xt - xb):
                v_lines.append((xt, xb, yl, yr))
            else:
                h_lines.append((xt, xb, yl, yr))

        plt.imshow(self.im_bin)

        for tpl in v_lines:
            xt, xb, yl, yr = tpl
            plt.plot([xt, xb], [yl, yr], c='r')

        for tpl in h_lines:
            xt, xb, yl, yr = tpl
            plt.plot([xt, xb], [yl, yr], c='g')

        plt.show()

        # Cleaning lines (if two are too close, we only keep the first one)
        v_lines = sorted(v_lines, key=lambda x: x[0])
        h_lines = sorted(h_lines, key=lambda x: x[2])

        v_lines = [v for i, v in enumerate(v_lines) if i == len(v_lines) - 1 or abs(v_lines[i + 1][0] - v[0]) > self.width / 25]  # 10
        h_lines = [h for i, h in enumerate(h_lines) if i == len(h_lines) - 1 or abs(h_lines[i + 1][2] - h[2]) > self.height / 25]  # 10

        # Completing lines (we need 10 * 10 lines)
        ii = 0
        while len(v_lines) < 10 and ii < 30:
            ii += 1
            avg_step = np.min(np.diff([v[0] for v in v_lines]))

            # Inserting before
            if v_lines[0][0] - avg_step > 0:
                v_lines.insert(0, (v_lines[0][0] - avg_step, v_lines[0][0] - avg_step, self.height, 0))

                # Inserting after
            if v_lines[-1][0] + avg_step < self.width:
                v_lines.insert(-1, (v_lines[-1][0] + avg_step, v_lines[-1][0] + avg_step, self.height, 0))

        ii = 0
        while len(h_lines) < 10 and ii < 30:
            ii += 1
            avg_step = np.min(np.diff([h[2] for h in h_lines]))

            # Inserting before
            if h_lines[0][2] - avg_step > 0:
                h_lines.insert(0, (self.width, 0, h_lines[0][2] - avg_step, h_lines[0][2] - avg_step))

                # Inserting after
            if h_lines[-1][2] + avg_step < self.height:
                h_lines.insert(-1, (self.height, 0, h_lines[-1][2] + avg_step, h_lines[-1][2] + avg_step))

        # plt.imshow(self.im_bin)
        #
        # for tpl in v_lines:
        #     xt, xb, yl, yr = tpl
        #     plt.plot([xt, xb], [yl, yr], c='r')
        #
        # for tpl in h_lines:
        #     xt, xb, yl, yr = tpl
        #     plt.plot([xt, xb], [yl, yr], c='g')
        #
        # plt.show()

        self.v_lines = sorted(v_lines, key=lambda x: x[0])
        self.h_lines = sorted(h_lines, key=lambda x: x[2])

    def detect_numbers(self):
        if self.h_lines is None:
            raise AttributeError('No grid extraction data found')

        index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        columns = list(range(9))
        self.sudoku_df = pd.DataFrame(index=index, columns=columns)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

        for hi in range(len(self.h_lines) - 1):
            for vi in range(len(self.v_lines) - 1):

                patch = copy(self.im_bin[self.h_lines[hi][2]:self.h_lines[hi + 1][2],
                             self.v_lines[vi][0]:self.v_lines[vi + 1][0]])
                #
                # patch[:10, :patch.shape[1]] = 0
                # patch[:patch.shape[0], :10] = 0
                # patch[-10:, :patch.shape[1]] = 0
                # patch[:patch.shape[0], -10:] = 0

                output = cv.connectedComponentsWithStats(patch, 4, cv.CV_32S)
                (numLabels, labels, stats, centroids) = output
                patch_center = [int(patch.shape[0] / 2), int(patch.shape[1] / 2)]
                labels_middle = labels[patch_center[0]-10:patch_center[0]+10, patch_center[1]-10:patch_center[1]+10]
                vv = [v for v in labels_middle.flatten() if v > 0]
                if len(vv) > 0:
                    label = np.median(vv)
                else:
                    label = -1

                patch = (labels == label).astype("uint8") * 255

                if patch.sum() > 20000 and patch.shape[0] > 20 and patch.shape[1] > 20:

                    # print(hi, vi)
                    # if hi == 1 and vi == 8:
                    #     plt.figure()
                    #     plt.imshow(patch)
                    #     plt.show()

                    max_sp = 0
                    detected = np.nan

                    im_crop = crop_number(patch)
                    x_proj, y_proj = get_projections(im_crop)

                    for i, proj_ref in self.refs.items():
                        x_ref, y_ref = proj_ref
                        x_proj_iter = interpolate(x_proj, x_ref)
                        y_proj_iter = interpolate(y_proj, y_ref)

                        x_proj_iter = x_proj_iter / np.linalg.norm(x_proj_iter)
                        y_proj_iter = y_proj_iter / np.linalg.norm(y_proj_iter)

                        sp_x = np.dot(x_ref, x_proj_iter)
                        sp_y = np.dot(y_ref, y_proj_iter)

                        sp = np.mean([sp_x, sp_y])

                        # if hi == vi == 2:
                        #     print(i, sp_x, sp_y, sp)

                        if sp > max_sp:
                            detected = i
                            max_sp = sp

                    if not np.isnan(detected):
                        self.sudoku_df.at[index[hi], vi] = detected
