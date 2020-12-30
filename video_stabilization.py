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
    image_process = ImageProcess()
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
                result, print_result = image_process.stabilize_picture(first_i, image2, {})

                ref_image = img_as_float(image_process.to_gray(image1))
                res_image = img_as_float(image_process.to_gray(result))

                print_result['score'] = ssim(ref_image, res_image, data_range=res_image.max() - res_image.min())
                print_results.append(print_result)

                print("'\rRemaining frames: {0}".format(frames - cnt), end='')

                vid_writer.write(result)
                image1 = np.copy(result)
        else:
            break

        cnt += 1

    print()
    print("Remaining frames: DONE")
    print("DURATION:", round(time.time() - start_time, 3), "s")

    # print_ordered("--result--", print_results)
    av = sum(item.get('score', 0) for item in print_results) / len(print_results)
    print("SSIM AVERAGE: ", round(av, 3))

    if capture.isOpened():
        capture.release()

    if vid_writer.isOpened():
        vid_writer.release()


if __name__ == '__main__':
    stabilize_video("images/Study_02_00007_01_L.avi")