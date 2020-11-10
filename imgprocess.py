from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


def fft_image(img_1, img_2):

    fft2 = np.fft.fft2(img_1)
    img_fft2 = np.log(1+np.abs(fft2))
    fft2_2 = np.fft.fft2(img_2)
    img_2_fft2 = np.log(1 + np.abs(fft2_2))

    fft_shift = np.fft.fftshift(fft2)
    img_fft_shift = np.log(1+np.abs(fft_shift))
    fft_shift_2 = np.fft.fftshift(fft2_2)
    img_2_fft_shift = np.log(1 + np.abs(fft_shift_2))

    fft_ishift = np.fft.ifftshift(fft_shift)
    img_fft_ishift = np.log(1+np.abs(fft_ishift))
    fft_ishift_2 = np.fft.ifftshift(fft_shift)
    img_2_fft_ishift = np.log(1 + np.abs(fft_ishift_2))

    # plt.subplot(421), plt.imshow(img_1, cmap='gray')
    # plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(422), plt.imshow(img_2, cmap='gray')
    # plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(423), plt.imshow(img_fft2, cmap='gray')
    # plt.title('After FFT 1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(424), plt.imshow(img_2_fft2, cmap='gray')
    # plt.title('After FFT 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(425), plt.imshow(img_fft_shift, cmap='gray')
    # plt.title('After FFT shift 1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(426), plt.imshow(img_2_fft_shift, cmap='gray')
    # plt.title('After FFT shift 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(427), plt.imshow(img_fft_ishift, cmap='gray')
    # plt.title('After FFT inverse shift 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(428), plt.imshow(img_2_fft_ishift, cmap='gray')
    # plt.title('After FFT inverse shift 2'), plt.xticks([]), plt.yticks([])
    # plt.show()

    phase = cv2.phaseCorrelate(np.float32(fft_ishift), np.float32(fft_ishift_2))
    (rows, cols) = img_2.shape[:2]
    res = cv2.warpAffine(img_2, phase, (rows, cols))
    plt.subplot(311), plt.imshow(img_1)
    plt.title("image 1"), plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(img_2)
    plt.title("image 2"), plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(res)
    plt.title("image 3"), plt.xticks([]), plt.yticks([])
    plt.show()


def phase_correlation(img1, img2):
    src1 = np.float32(img1)
    src2 = np.float32(img2)
    print(cv2.phaseCorrelate(src1, src2))


if __name__ == '__main__':
    img_1 = cv2.imread("images/retina.jpg", 0)
    img_2 = cv2.imread("images/retina_posun.jpg", 0)
    fft_image(img_1, img_2)
    # phase_correlation(img_1, img_2)
