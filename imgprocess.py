from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


def fft_image():
    print("FTT of image.")
    img = cv2.imread("img.jpg", 0)
    f = np.fft.fft2(img)

    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:, :, 0] + 1j * f_shift[:, :, 1]
    f_abs = np.abs(f_complex) + 1  # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(f_img, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    fft_image()
