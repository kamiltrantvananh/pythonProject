import numpy as np
import cv2
from matplotlib import pyplot as plt
from vidstab import VidStab
import collections
import math
from sklearn.metrics import jaccard_score
import time


def stabilize_video(f_name):
    """
    Stabilization of video sample.

    :param f_name: path to the video sample file
    :return: stabilized video
    """
    capture = cv2.VideoCapture(f_name)

    # load camera
    frsize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = capture.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter("stabilized.avi", fourcc, fps, frsize)

    print("Start: ")
    first = True
    cnt = 0
    image1 = None
    print_results = []

    while True:
        ret, image2 = capture.read()

        if ret:
            # read the camera image:
            if first:
                first = False
                image1 = np.copy(image2)
            else:
                # perform stabilization
                result, print_result = stabilize_picture(image1, image2, {})
                score = jaccard_score(image1.flatten(), result.flatten(), average='macro')
                print_result['score'] = score
                print_results.append(print_result)
                vid_writer.write(result)
                image1 = np.copy(result)
        else:
            break

        print("frame: " + str(cnt), end='\r')
        cnt += 1

    print("Done")
    print_ordered("--result--", print_results)

    if capture.isOpened():
        capture.release()

    if vid_writer.isOpened():
        vid_writer.release()


def stabilize_picture(_img1, _img2, print_result=None):
    """
    Stabilization of two pictures.

    :param print_result: for print results
    :param _img1: first imgge
    :param _img2: second image
    :return:
    """
    if print_result is None:
        print_result = {}
    img1, img1_gray, img1_polar = process_image(_img1)
    img2, img2_gray, img2_polar = process_image(_img2)

    (rows, cols) = img1_gray.shape

    result_image, print_result = shift_stabilization(img1_gray, img2_gray, img2, rows, cols, print_result)
    result_image, print_result = rotation_scale_stabilization(img1_polar, img2_polar, result_image, rows, cols, print_result)

    return result_image, print_result


def rotation_scale_stabilization(img1_polar, img2_polar, img2_to_stabilized, rows, cols, print_result):
    (log_polar_cx, log_polar_cy), _ = cv2.phaseCorrelate(np.float32(img1_polar), np.float32(img2_polar))
    rotation, scale = scale_rotation(log_polar_cy, rows, cols)
    print_result['scale'] = scale
    print_result['rotation'] = rotation
    centre = (cols//2, rows//2)
    transformation_matrix = cv2.getRotationMatrix2D(centre, rotation, 1)
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    result_image = cv2.warpAffine(img2_to_stabilized, transformation_matrix, dsize=(cols, rows), flags=flags)
    return result_image, print_result


def shift_stabilization(img1_gray, img2_gray, img2_to_stabilized, rows, cols, print_result=None):
    hanning = cv2.createHanningWindow((cols, rows), cv2.CV_32F)
    (cx, cy), _ = cv2.phaseCorrelate(np.float32(img1_gray), np.float32(img2_gray), window=hanning)
    (cx, cy) = (round(cx, 2), round(cy, 2))
    M = np.float32([[1, 0, cx], [0, 1, cy]])
    print_result['x'] = cx
    print_result['y'] = cy
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    result_image = cv2.warpAffine(img2_to_stabilized, M, dsize=(cols, rows), flags=flags)
    return result_image, print_result


def scale_rotation(cy, rows, cols):
    """
    Compute angle and scale of the point based on Cartesian coordinate system.

    :param cy: base y
    :param rows: length of the picture
    :param cols: width of the picture
    :return: angle and scale
    """
    rotation = -cy / rows * 360
    rotation = round(rotation, 1)

    scale = math.exp(math.log(rows * 1.1 / 2.0) / max(rows, cols))
    scale = 1.0 / math.pow(scale, cy)
    scale = round(scale)

    return rotation, scale


def process_image(img):
    """
    Apply RGB to gray scale image and transform using log polar transformation.

    :param img: image source
    :return: original image, gray scale image, log polar image
    """

    # gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # log polar image
    (rows, cols) = img_gray.shape
    center = (cols // 2, rows // 2)
    M = rows / (math.log(round(min(rows, cols) / 2)))
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    img_polar = cv2.logPolar(np.float32(img_gray), center, M, flags)
    # img_polar = cv2.linearPolar(np.float32(img_gray), center, min(rows, cols), 0)

    return img, img_gray, img_polar


def existing_solution():
    """
    Existing solution of stabilization video sample.

    :return: stabilized video
    """
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='images/Study_02_00014_01_R.avi', output_path='stable_video.avi')


def stabilize_images(img1, img2, print_result=None):
    res, print_result = stabilize_picture(img1, img2, print_result)

    # For testing purpose
    plt.subplot(311), plt.imshow(img1)
    plt.title("image 1"), plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(img2)
    plt.title("image 2"), plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(res)
    plt.title("result image"), plt.xticks([]), plt.yticks([])
    plt.show()
    return res, print_result


def print_ordered(key_text, dict_values):
    print()
    print(key_text)

    if isinstance(dict_values, list):
        for i in dict_values:
            _print_ordered(i)
    else:
        _print_ordered(dict_values)


def _print_ordered(dict_values):
    ordered = collections.OrderedDict(sorted(dict_values.items()))
    for k, v in ordered.items():
        print(k, v)


def print_score(reg_img, res_img):
    score = jaccard_score(reg_img.flatten(), res_img.flatten(), average='macro')
    print()
    print("SCORE: ", round(score, 5))
    print()


def test_shift_image(img):
    num_rows, num_cols = img.shape[:2]
    expected_values = {
        'x': 25,
        'y': 25,
        'rotation': 45,
        'scale': 1
    }
    print_ordered("--expect--", expected_values)

    translation_matrix = np.float32([[1, 0, expected_values['x']], [0, 1, expected_values['y']]])
    img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    centre = tuple(np.array(img.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, expected_values['rotation'], expected_values['scale'])
    img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

    return img


def test_stabilize_two_images():
    img1 = cv2.imread("images/retina.jpg")
    img2 = test_shift_image(img1)
    # img2 = cv2.imread("images/retina_posun.jpg")
    curr_time = time.time()
    res, print_values = stabilize_images(img1, img2)
    time_duration = time.time() - curr_time
    print("DURATION: ", round(time_duration, 3), "ms")
    print_ordered("--result--", print_values)
    print_score(img1, res)


def main():
    # test_stabilize_two_images()
    stabilize_video("images/Study_02_00007_01_L.avi")

    # existing_solution()


if __name__ == '__main__':
    main()
