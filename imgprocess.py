import numpy as np
import cv2
from matplotlib import pyplot as plt
import collections
import math
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import time


def get_video_writer(video_name, frame_size, fps):
    """
    Get video writer for stabilized video.

    :param video_name:  name of the stabilized video
    :param frame_size:  frame size
    :param fps:         frame per seconds
    :return: video writer class from OpenCV, see <https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html>
    """
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    return cv2.VideoWriter(video_name, fourcc, fps, frame_size)


def stabilize_video(f_name):
    """
    Stabilization of video sample.

    :param f_name:  path to the video sample file
    """
    capture = cv2.VideoCapture(f_name)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = capture.get(cv2.CAP_PROP_FPS)
    vid_writer = get_video_writer(f_name + "_stabilized.avi", frame_size, fps)

    print("Start stabilization of video", f_name)
    cnt = 0
    print_results = []
    first = True
    first_i = None
    image1 = None

    start_time = time.time()
    while True:
        ret, image2 = capture.read()

        if ret:
            # read the camera image:
            if first:
                first = False
                image1 = np.copy(image2)
                first_i = image1
            else:
                # perform stabilization
                result, print_result = stabilize_picture(first_i, image2, {})

                ref_image = img_as_float(to_gray(image1))
                res_image = img_as_float(to_gray(result))

                print_result['score'] = ssim(ref_image, res_image, data_range=res_image.max() - res_image.min())
                print_results.append(print_result)

                if cnt % 10 == 0:
                    print("\rRemain frames:", frames - cnt)

                vid_writer.write(result)
                image1 = np.copy(result)
        else:
            break

        cnt += 1

    print("DONE")
    print("DURATION:", round(time.time() - start_time, 3), "s")

    # print_ordered("--result--", print_results)
    av = sum(item.get('score', 0) for item in print_results) / len(print_results)
    print("SSIM AVERAGE: ", round(av, 3))

    if capture.isOpened():
        capture.release()

    if vid_writer.isOpened():
        vid_writer.release()


def stabilize_picture(image_reference, image_target, print_result=None):
    """
    Stabilization of two pictures.

    :param print_result:        gathered information for stabilization process
    :param image_reference:     reference image
    :param image_target:        target image
    :return:
        result_image:   stabilized target image
        print_result:   collected information during stabilization process
    """
    if print_result is None:
        print_result = {}

    if image_reference.shape != image_target.shape:
        raise NameError("Reference image and target image have different shapes!",
                        image_reference.shape, image_target.shape)

    (rows, cols) = to_gray(image_reference).shape

    result_image, print_result = shift_stabilization(
        to_gray(image_reference),
        to_gray(image_target),
        image_target,
        rows,
        cols,
        print_result
    )

    result_image, print_result = rotation_scale_stabilization(
        to_log_polar(to_gray(image_reference)),
        to_log_polar(to_gray(result_image)),
        result_image,
        rows,
        cols,
        print_result
    )

    return result_image, print_result


def rotation_scale_stabilization(img1_polar, img2_polar, img2_to_stabilized, rows, cols, print_result):
    """
    Perform rotation and scale stabilization using phase correlation on two log polar images.

    :param img1_polar:
    :param img2_polar:
    :param img2_to_stabilized:
    :param rows:
    :param cols:
    :param print_result:
    :return:
    """
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
    """
    Perform shift stabilization on two images using phase correlation with hanning window

    :param img1_gray:           gray scale source image
    :param img2_gray:           gray scale target image
    :param img2_to_stabilized:  image to be stabilized
    :param rows:                rows of result image
    :param cols:                columns of result image
    :param print_result:        gathered information during stabilization
    :return:
        result_image:   stabilized (shifted) image
        print_result:   collected information during shift stabilization
    """
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

    :param cy:      base y
    :param rows:    length of the picture
    :param cols:    width of the picture
    :return:
        rotation:   difference angle in degrees
        scale:      difference scale
    """
    rotation = -cy / rows * 360
    rotation = round(rotation, 1)

    scale = math.exp(math.log(rows * 1.1 / 2.0) / max(rows, cols))
    scale = 1.0 / math.pow(scale, cy)
    scale = round(scale)

    return rotation, scale


def process_image(image):
    """
    Perform image to gray scale and processed log polar transformation.

    :param image: image source
    :return:
        image:          original image
        image_gray:     gray scale image
        image_polar:    log polar image
    """

    # gray image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # log polar image
    (rows, cols) = image_gray.shape
    center = (cols // 2, rows // 2)
    M = rows / (math.log(round(min(rows, cols) / 2)))
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    image_polar = cv2.logPolar(np.float32(image_gray), center, M, flags)

    return image, image_gray, image_polar


def to_log_polar(image_gray):
    """
    Convert gray scale image to log polar image.

    :param image_gray: target gray scale image
    :return: log polar image
    """
    (rows, cols) = image_gray.shape
    center = (cols // 2, rows // 2)
    M = rows / (math.log(round(min(rows, cols) / 2)))
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    return cv2.logPolar(np.float32(image_gray), center, M, flags)


def to_gray(image):
    """
    Convert image to gray scale.

    :param image: target image
    :return: gray scale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def stabilize_images(image_source, image_target, print_result=None):
    """
    Function for testing purpose to stabilize images using phase correlation. Also it will show result image using
    "pyplot" library.

    :param image_source: source image which will be reference
    :param image_target: image for stabilization
    :param print_result: gathered information about stabilization
    :return:
        res:            stabilized image
        print_result:   collected information during stabilization
    """
    res, print_result = stabilize_picture(image_source, image_target, print_result)

    # pyplot
    plt.subplot(311), plt.imshow(image_source)
    plt.title("image 1"), plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(image_target)
    plt.title("image 2"), plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(res)
    plt.title("result image"), plt.xticks([]), plt.yticks([])
    plt.show()
    return res, print_result


def print_ordered(key_text, dict_values):
    """
    Print collected informations during stabilization process.

    :param key_text:        label
    :param dict_values:     collected information in array of dictionary or only dictionary
    """
    print()
    print(key_text)

    if isinstance(dict_values, list):
        for i in dict_values:
            _print_ordered(i)
    else:
        _print_ordered(dict_values)


def _print_ordered(dict_values):
    """
    Sort and print content of input dictionary.

    :param dict_values: collected information i n dictionary
    """
    ordered = collections.OrderedDict(sorted(dict_values.items()))
    for k, v in ordered.items():
        print(k, v)


def print_score(ref_img, res_img):
    """
    Compute Jaccard score on two images and print a result.

    :param ref_img: reference image
    :param res_img: result image
    """
    # score = jaccard_score(ref_img.flatten(), res_img.flatten(), average='macro')
    ref_img_float = img_as_float(to_gray(ref_img))
    res_img_float = img_as_float(to_gray(res_img))
    score = ssim(ref_img_float, res_img_float, data_range=res_img_float.max() - res_img_float.min())
    print("-------------------------")
    print("SSIM: ", round(score, 2),)
    print("-------------------------")


def _test_transform_image(image_target):
    """
    Function for transformation of image (rotation, scale and shift) for test purpose.

    :param image_target: target image to transform
    :return: transformed image
    """
    num_rows, num_cols = image_target.shape[:2]
    expected_values = {
        'x': 25,
        'y': 25,
        'rotation': 45,
        'scale': 1
    }
    print_ordered("--expect--", expected_values)

    translation_matrix = np.float32([[1, 0, expected_values['x']], [0, 1, expected_values['y']]])
    image_target = cv2.warpAffine(image_target, translation_matrix, (num_cols, num_rows))

    centre = tuple(np.array(image_target.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, expected_values['rotation'], expected_values['scale'])
    image_target = cv2.warpAffine(image_target, rotation_matrix, (num_cols, num_rows))

    return image_target


def _test_stabilize_two_images():
    """
    Test function for stabilize two images and print collected data during stabilization.
    """
    # Given
    img1 = cv2.imread("images/retina.jpg")
    img2 = _test_transform_image(img1)
    # img2 = cv2.imread("images/retina_posun.jpg")

    # When
    curr_time = time.time()
    res, print_values = stabilize_images(img1, img2)
    time_duration = time.time() - curr_time

    # Then
    print("~TIME DURATION: ", round(time_duration, 3), "s")
    print_ordered("--result--", print_values)
    print_score(img1, res)


def main():
    # _test_stabilize_two_images()
    stabilize_video("images/Study_02_00007_01_L.avi")


if __name__ == '__main__':
    main()
