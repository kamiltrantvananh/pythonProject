import numpy as np
import cv2
import math
import collections
from skimage import img_as_float, img_as_ubyte
from skimage import transform
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


class ImageProcess(object):
    """
    This class provides methods for stabilize 2 images by using phase correlation
    """
    def stabilize_picture(self, image_reference, image_target, print_result=None):
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

        (rows, cols) = self.to_gray(image_reference).shape

        result_image, print_result = self.shift_stabilization(
            self.to_gray(image_reference),
            self.to_gray(image_target),
            image_target,
            rows,
            cols,
            print_result
        )

        result_image, print_result = self.rotation_scale_stabilization(
            self.to_log_polar(self.to_gray(image_reference)),
            self.to_log_polar(self.to_gray(result_image)),
            result_image,
            rows,
            cols,
            print_result
        )

        return result_image, print_result

    @staticmethod
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
        rotation, scale = ImageProcess.__scale_rotation(log_polar_cy, rows, cols)
        print_result['scale'] = scale
        print_result['rotation'] = rotation
        centre = (cols // 2, rows // 2)
        transformation_matrix = cv2.getRotationMatrix2D(centre, rotation, 1)
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        result_image = cv2.warpAffine(img2_to_stabilized, transformation_matrix, dsize=(cols, rows), flags=flags)
        return result_image, print_result

    @staticmethod
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
        t_form = transform.EuclideanTransform(translation=(cx, cy))
        result_image = transform.warp(img2_to_stabilized, t_form)
        return img_as_ubyte(result_image), print_result

    @staticmethod
    def __scale_rotation(cy, rows, cols):
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
        # rotation = round(rotation, 1)

        scale = math.exp(math.log(rows * 1.1 / 2.0) / max(rows, cols))
        scale = 1.0 / math.pow(scale, cy)
        scale = round(scale)

        return rotation, scale

    @staticmethod
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

    @staticmethod
    def to_gray(image):
        """
        Convert image to gray scale.

        :param image: target image
        :return: gray scale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
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
                ImageProcess.__print_ordered(i)
        else:
            ImageProcess.__print_ordered(dict_values)

    @staticmethod
    def __print_ordered(dict_values):
        """
        Sort and print content of input dictionary.

        :param dict_values: collected information i n dictionary
        """
        ordered = collections.OrderedDict(sorted(dict_values.items()))
        for k, v in ordered.items():
            print(k, v)

    @staticmethod
    def print_score(ref_img, res_img):
        """
        Compute SSIM score on two images and print a result.

        :param ref_img: reference image
        :param res_img: result image
        """
        # score = jaccard_score(ref_img.flatten(), res_img.flatten(), average='macro')
        ref_img_float = img_as_float(ImageProcess.to_gray(ref_img))
        res_img_float = img_as_float(ImageProcess.to_gray(res_img))
        score = ssim(ref_img_float, res_img_float, data_range=res_img_float.max() - res_img_float.min(),
                     gaussian_weights=True)
        print("-------------------------")
        print("SSIM: ", round(score, 2), )
        print("-------------------------")

    @staticmethod
    def rmse(ref_img, res_img):
        return mean_squared_error(ref_img, res_img, squared=False)
