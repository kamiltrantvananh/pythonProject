import numpy as np
import cv2
from matplotlib import pyplot as plt
from vidstab import VidStab
import sys
import math


def stabilize_video(f_name):
    """
    Stabilization of video sample.

    :param f_name: path to the video sample file
    :return: stabilized video
    """
    capture = cv2.VideoCapture(f_name)

    # load camera
    frsize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = capture.get(cv2.CAP_PROP_FPS)
    vidWriter = cv2.VideoWriter("stabilized.avi", fourcc, fps, frsize, False)

    print("Start: ")
    first = True
    cnt = 0
    image2 = None

    while True:
        ret, image1 = capture.read()

        if ret:
            # read the camera image:
            if first:
                # vidWriter.write(image1)
                first = False
            else:
                # perform stabilization
                result = stabilize_picture(image1, image2)
                vidWriter.write(result)

            image2 = np.copy(image1)
        else:
            break

        sys.stdout.write("\rRemaining:" + str(fps - cnt))
        sys.stdout.flush()
        cnt += 1

    sys.stdout.write("\rDone")

    if vidWriter.isOpened():
        vidWriter.release()


def stabilize_picture(_img1, _img2):
    """
    Stabilization of two pictures.

    :param _img1: first imgge
    :param _img2: second image
    :return:
    """
    img1, img1_gray, img1_polar = process_image(_img1)
    img2, img2_gray, img2_polar = process_image(_img2)

    rows, cols = np.size(img1, axis=1), np.size(img1, axis=0)
    rotated_img = rotation_scale_stabilization(img1_polar, img2_polar, img2_gray, rows, cols)
    result_image = shift_stabilization(img1_gray, img2_gray, rotated_img, rows, cols)

    # For testing purpose
    # plt.subplot(411), plt.imshow(img1_gray)
    # plt.title("image 1"), plt.xticks([]), plt.yticks([])
    # plt.subplot(412), plt.imshow(img2_gray)
    # plt.title("image 2"), plt.xticks([]), plt.yticks([])
    # plt.subplot(413), plt.imshow(rotated_img)
    # plt.title("rotated image"), plt.xticks([]), plt.yticks([])
    # plt.subplot(414), plt.imshow(result_image)
    # plt.title("result_image"), plt.xticks([]), plt.yticks([])
    # plt.show()

    return result_image


def rotation_scale_stabilization(img1_polar, img2_polar, img1_gray, rows, cols):
    """


    :param img1_polar:
    :param img2_polar:
    :param img1_gray:
    :param rows:
    :param cols:
    :return:
    """
    (log_polar_cx, log_polar_cy), _ = cv2.phaseCorrelate(np.float32(img2_polar), np.float32(img1_polar))
    # print(str(log_polar_cx) + " " + str(log_polar_cy))
    # (a, b), _ = cv2.phaseCorrelate(np.float32(img2_polar), np.float32(img1_polar))
    # print(str(a) + " " + str(b))
    rotation, scale = scale_rotation(log_polar_cx, log_polar_cy, rows, cols)
    transformation_matrix = cv2.getRotationMatrix2D((rows / 2, cols / 2), rotation, 1)
    return cv2.warpAffine(img1_gray, transformation_matrix, dsize=(rows, cols), flags=cv2.INTER_CUBIC)


def shift_stabilization(img1_gray, img2_gray, rotated_img, rows, cols):
    (cx, cy), _ = cv2.phaseCorrelate(np.float32(img2_gray), np.float32(img1_gray))
    M = np.float32([[1, 0, cx], [0, 1, cy]])
    return cv2.warpAffine(rotated_img, M, dsize=(rows, cols), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def wrap_angle(angles, ceil=2 * np.pi):
    """
    Args:
        angles (float or ndarray, unit depends on kwarg ``ceil``)
        ceil (float): Turnaround value
    """
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def scale_rotation(cx, cy, rows, cols):
    """
    Compute angle and scale of the point based on Cartesian coordinate system.

    :param cx: base x
    :param cy: base y
    :param rows: length of the picture
    :param cols: width of the picture
    :return: angle and scale
    """
    rotation = -np.pi * cx / float(rows)
    rotation = rotation * 180.0 / np.pi
    rotation += 180.0
    rotation = np.fmod(rotation, 360.0)
    rotation -= 180.0
    rotation = -rotation

    scale = math.exp(math.log(rows * 1.1 / 2.0) / max(rows, cols))
    scale = 1.0 / math.pow(scale, cy)

    return rotation, scale


def process_image(img):
    """
    Apply RGB to gray scale image and transform using log polar transformation.

    :param img: image source
    :return: original image, gray scale image, log polar image
    """

    # gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # log polar image
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = h / (math.log(round(min(h, w) / 2)))
    img_polar = cv2.logPolar(img_gray, center, M, cv2.WARP_FILL_OUTLIERS)

    return img, img_gray, img_polar


def existing_solution():
    """
    Existing solution of stabilization video sample.

    :return: stabilized video
    """
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='images/Study_02_00014_01_R.avi', output_path='stable_video.avi')


if __name__ == '__main__':
    stabilize_video("images/Study_02_00007_01_L.avi")

    # img1 = cv2.imread("images/retina.jpg")
    # img2 = cv2.imread("images/retina_rotacia.jpg")
    # stabilize_picture(img1, img2)

    # existing_solution()
