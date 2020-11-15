import numpy as np
import cv2
from matplotlib import pyplot as plt
from vidstab import VidStab


def fft_image(image_1, image_2):
    """
    Stabilization two images using phase correlation.

    :param image_1: first image
    :param image_2: second image
    :return: show result of stabilization
    """
    (sx, sy), _ = cv2.phaseCorrelate(np.float32(image_1), np.float32(image_2))
    cols, rows = np.size(image_1, axis=1), np.size(image_1, axis=0)
    trans_matrix = np.float32([[1, 0, sx], [0, 1, sy]])
    res = cv2.warpAffine(image_2, M=trans_matrix, dsize=(cols, rows))

    # plt.subplot(131), plt.imshow(img_1)
    # plt.title("image 1"), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(img_2)
    # plt.title("image 2"), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(res)
    # plt.title("result"), plt.xticks([]), plt.yticks([])

    # plt.imshow(img_1, alpha=100)
    plt.imshow(image_2, alpha=90)
    plt.imshow(res, alpha=70)

    plt.show()


def video_stabilization(f_name):
    """
    Stabilization of video sample.
    :param f_name: path to the video sample file
    :return: stabilized video
    """
    capture = cv2.VideoCapture(f_name)

    # load camera
    frsize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vidWriter = cv2.VideoWriter("stabilized.wmv", cv2.VideoWriter_fourcc(*'MJPG'), capture.get(cv2.CAP_PROP_FPS),
                                frsize)
    print(vidWriter)

    first = True
    shift = np.array([0, 0])
    # cx = 0.0
    # cy = 0.0
    while True:
        ret, img1 = capture.read()
        if ret:
            # read the camera image:
            if first:
                first = False
            else:
                imgNew = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                imgLast = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                # perform stabilization
                reet = cv2.phaseCorrelate(np.float32(imgNew), np.float32(imgLast))

                alpha = 0.33333
                shift[0] = alpha * float(reet[0][0]) + (1 - alpha) * float(shift[0])
                shift[1] = alpha * float(reet[0][1]) + (1 - alpha) * float(shift[1])

                # shift the new image
                trans_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

                # shift the new image
                # cx = cx - reet[0][0]
                # cy = cy - reet[0][1]
                cols, rows = np.size(img1, axis=1), np.size(img1, axis=0)

                img3 = cv2.warpAffine(img2, M=trans_matrix, dsize=(cols, rows),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # export image
                vidWriter.write(img3)
            img2 = np.copy(img1)
        else:
            break

    if vidWriter.isOpened():
        vidWriter.release()


def existing_solution():
    """
    Existing solution of stabilization video sample.
    :return: stabilized video
    """
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='images/Study_02_00007_01_L.avi', output_path='stable_video.avi')


if __name__ == '__main__':
    img_1 = cv2.imread("images/retina_posun.jpg")
    img_2 = cv2.imread("images/retina.jpg")
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    fft_image(img_1, img_2)

    # video_stabilization("images/Study_02_00007_01_L.avi")

    # existing_solution()
