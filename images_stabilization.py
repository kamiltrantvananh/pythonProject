import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from img_process import ImageProcess


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
    image_process = ImageProcess()
    res, print_result = image_process.stabilize_picture(image_source, image_target, print_result)

    # pyplot
    plt.subplot(311), plt.imshow(image_source)
    plt.title("image 1"), plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(image_target)
    plt.title("image 2"), plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(res)
    plt.title("result image"), plt.xticks([]), plt.yticks([])
    plt.show()
    return res, print_result


def _test_transform_image(image_target):
    """
    Function for transformation of image (rotation, scale and shift) for test purpose.

    :param image_target: target image to transform
    :return: transformed image
    """
    num_rows, num_cols = image_target.shape[:2]
    expected_values = {
        'x': 50,
        'y': 50,
        'rotation': 45,
        'scale': 1
    }
    ImageProcess.print_ordered("--expect--", expected_values)

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
    img = cv2.imread("images/retina.jpg")
    img2 = _test_transform_image(img)

    # When
    curr_time = time.time()
    res, print_values = stabilize_images(img, img2)
    time_duration = time.time() - curr_time

    # Then
    print("~TIME DURATION: ", round(time_duration, 3), "s")
    ImageProcess.print_ordered("--result--", print_values)
    ImageProcess.print_score(img, res)


if __name__ == '__main__':
    _test_stabilize_two_images()
