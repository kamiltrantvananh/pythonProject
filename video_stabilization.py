from img_process import ImageProcess
import time
import cv2
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim


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


def stabilize_video(f_name, use_first_as_reference=False, print_full_results=False, use_gaussian_weights=True):
    """
    Stabilization of video sample.

    :param f_name:                  path to the video sample file
    :param use_first_as_reference:  use first frame as a reference image, default is False
    :param print_full_results:      if you want print all results from stabilization
    :param use_gaussian_weights:    use gaussian weights in SSIM calculation
    """
    capture = cv2.VideoCapture(f_name)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = capture.get(cv2.CAP_PROP_FPS)
    stabilized_video_name = f_name + "_stabilized.avi"
    vid_writer = get_video_writer(stabilized_video_name, frame_size, fps)

    print("Start stabilization of video:", f_name)
    print("Using first frame as reference:", use_first_as_reference)
    print("Print all stabilization results:", print_full_results)
    print("Using Gaussian weights for SSIM score:", use_gaussian_weights)

    image_process = ImageProcess()
    cnt = 0
    print_results = []
    first = True
    first_i, image1 = None, None

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
                if use_first_as_reference:
                    result, print_result = image_process.stabilize_picture(first_i, image2)
                    ref_image = img_as_float(image_process.to_gray(first_i))
                else:
                    result, print_result = image_process.stabilize_picture(image1, image2)
                    ref_image = img_as_float(image_process.to_gray(image1))

                # write stabilized frame
                vid_writer.write(result)
                image1 = np.copy(result)

                # compute SSIM index
                res_image = img_as_float(image_process.to_gray(result))
                print_result['score'] = ssim(ref_image,
                                             res_image,
                                             data_range=res_image.max() - res_image.min(),
                                             gaussian_weights=use_gaussian_weights)
                print_results.append(print_result)
                print("'\rRemaining frames: {0}".format(frames - cnt), end='')
        else:
            break

        cnt += 1

    if capture.isOpened():
        capture.release()

    if vid_writer.isOpened():
        vid_writer.release()

    print("'\rRemaining frames: DONE", end='\n')
    print("Result stabilized video:", stabilized_video_name)
    print("-STATS--------------")
    print("DURATION:", round(time.time() - start_time, 3), "s")

    if print_full_results:
        ImageProcess.print_ordered("ALL RESULTS::", print_results)

    av = sum(item.get('score', 0) for item in print_results) / len(print_results)
    print("SSIM AVERAGE: ", round(av, 3))


if __name__ == '__main__':
    stabilize_video("images/Study_02_00007_01_L.avi", use_first_as_reference=True)
