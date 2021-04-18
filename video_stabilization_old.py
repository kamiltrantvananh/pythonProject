import getopt
import sys
import time
from pandas import *
import cv2
import numpy as np
from skimage import img_as_float
from matplotlib import pyplot as plt

from img_process_old import ImageProcess


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


def stabilize_video(f_name, use_first_as_reference=False, print_full_results=False, use_gaussian_weights=True,
                    selected_points=None):
    """
    Stabilization of video sample.

    :param f_name:                  path to the video sample file
    :param use_first_as_reference:  use first frame as a reference image, default is False
    :param print_full_results:      if you want print all results from stabilization
    :param use_gaussian_weights:    use gaussian weights in SSIM calculation
    :param selected_points:         selected 5 points
    """
    if selected_points is None:
        selected_points = {}
    capture = cv2.VideoCapture(f_name)

    CROP_X = 100
    CROP_Y = 100

    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps = capture.get(cv2.CAP_PROP_FPS)
    stabilized_video_name = f_name + "_stabilized.avi"
    vid_writer = get_video_writer(stabilized_video_name, frame_size, fps)

    print("Start stabilization of video:", f_name)
    print("Using first frame as reference:", use_first_as_reference)

    image_process = ImageProcess()

    r_frames = 0
    print_results = []
    first = True
    first_i, image1 = None, None

    euclid_distances = []

    start_time = time.time()
    while True:
        ret, image2 = capture.read()

        if ret:
            # read the camera image:
            rows, cols, _ = image2.shape
            if first:
                first = False
                image1 = np.copy(image2)
                first_i = image1
                if not selected_points:
                    selected_points = image_process.select_reference_points(image2)
            else:
                # perform stabilization
                if use_first_as_reference:
                    result, print_result = image_process.stabilize_picture(first_i, image2, crop_sizes=(CROP_X, CROP_Y))
                    ref_image = img_as_float(image_process.to_gray(first_i))
                else:
                    result, print_result = image_process.stabilize_picture(image1, image2)
                    ref_image = img_as_float(image_process.to_gray(image1))

                if r_frames % 10 == 0:
                    tracking_points = image_process.tracking_points(selected_points, result)
                    euclid_distances.append(image_process.euclid_distance(selected_points, tracking_points))

                # write stabilized frame
                vid_writer.write(result)
                image1 = np.copy(result)

                # statistics
                print_results.append(print_result)
                print("'\rRemaining frames: {0}. Actual frame: {1}".format(frames - r_frames, r_frames), end='')
        else:
            break

        r_frames += 1

    if capture.isOpened():
        capture.release()

    if vid_writer.isOpened():
        vid_writer.release()

    print("'\rRemaining frames: DONE", end='\n')
    print("Result stabilized video:", stabilized_video_name)
    print("-STATS--------------")
    print("DURATION:", round(time.time() - start_time, 3), "s")

    print("EUCLID DISTANCES: ", euclid_distances)
    euclid_average = mean_euclid_distances(euclid_distances)
    print("EUCLID DISTANCES AVERAGE: ", euclid_average)

    if print_full_results:
        image_process.print_ordered("ALL RESULTS::", print_results)

    print("STD: ", std_euclids(euclid_distances, euclid_average))
    boxplot_euclids(euclid_distances)
    print()


def boxplot_euclids(euclid_distances):
    euclids = get_euclid_distance_per_point(euclid_distances)
    plt.boxplot(euclids, notch=True)
    plt.xlabel("Vybrané body (-)")
    plt.ylabel("Euklidovská vzdialenosť (-)")
    plt.title("Variabilita euklidovských vzdialeností vybraných bodov")
    plt.show()
    plt.savefig('Euclid distances.png')


def get_euclid_distance_per_point(euclid_distances):
    euclids = []
    for i in range(len(euclid_distances[0])):
        euclid = []
        for euclid_distance in [elem[i] for elem in euclid_distances]:
            if euclid_distance != -1:
                euclid.append(euclid_distance)
        euclids.append(euclid)

    return euclids


def mean_euclid_distances(euclid_distances):
    averages = []
    for i in range(len(euclid_distances[0])):
        tmp = 0
        cnt = 0
        for euclid_distance in [elem[i] for elem in euclid_distances]:
            if euclid_distance != -1:
                tmp += euclid_distance
                cnt += 1
        averages.append(tmp/cnt) if cnt != 0 else tmp

    return averages


def std_euclids(euclid_distances, mean):
    var = []
    for i in range(len(euclid_distances[0])):
        tmp = 0
        cnt = 0
        for euclid_distance in [elem[i] for elem in euclid_distances]:
            if euclid_distance != -1:
                tmp += (euclid_distance - mean)**2
                cnt += 1
        var.append(tmp/cnt) if cnt != 0 else tmp

    return np.sqrt(var)


def main(argv):
    if len(argv) == 0:
        raise NameError("Missing video sample file!")

    opts, file_paths = getopt.getopt(argv, "fs:", ["use-first-as-reference", "selected-points"])
    first_as_reference = False
    selected_points = {}
    for opt, arg in opts:
        if opt in ('-f', '--use-first-as-reference'):
            first_as_reference = True
        if opt in ('-s', '--selected-points'):
            excel = pandas.read_excel(arg, index_col=0).to_dict()["value"]
            for k in excel:
                x, y = tuple(map(int, k.split(',')))
                selected_points[(x, y)] = excel[k]

    if len(file_paths) == 0:
        raise ValueError("Missing path to video sample.")

    print(selected_points)
    for file_path in file_paths:
        stabilize_video(file_path, use_first_as_reference=first_as_reference, selected_points=selected_points)

# mean image as a reference
#def mean_image(f_name, frames):
#    capture = cv2.VideoCapture(f_name)
#    while True:
#        ret, image = capture.read()
#        if ret:
#            # is there
#            np.mean(image)/frames
#        else:
#            break
#    return ref_image = np.mean(image)/frames

if __name__ == '__main__':
    main(sys.argv[1:])

