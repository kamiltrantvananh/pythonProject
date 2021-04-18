import numpy as np
import cv2
import math
import collections
from skimage import img_as_float, img_as_ubyte
from skimage import transform
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from scipy.spatial import distance
from scipy.fftpack import fft2


class ImageProcess(object):
    """
    This class provides methods for stabilize 2 images by using phase correlation
    """

    def stabilize_picture(self, image_reference, image_target, print_result=None, crop_sizes=(0, 0)):
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
            image_reference,
            image_target,
            print_result,
            crop_sizes
        )

        result_image, print_result = self.rotation_scale_stabilization(
            image_reference,
            result_image,
            print_result,
            crop_sizes
        )

        return result_image, print_result

    @staticmethod
    def rotation_scale_stabilization(img1, img2, print_result, crop_sizes):
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
        radius = 420

        img1_gray = ImageProcess.crop_image(ImageProcess.to_gray(img1), crop_sizes)
        m1 = np.abs(fft2(img1_gray))
        img1_polar = ImageProcess.to_log_polar(m1)
        shape = img1_gray.shape
        img1_polar = img1_polar[:shape[0] // 2, :]

        img2_gray = ImageProcess.crop_image(ImageProcess.to_gray(img2), crop_sizes)
        m2 = np.abs(fft2(img2_gray))
        img2_polar = ImageProcess.to_log_polar(m2)
        img2_polar = img2_polar[:shape[0] // 2, :]

        (rows, cols) = img1_gray.shape

        (log_polar_cx, log_polar_cy), _ = cv2.phaseCorrelate(np.float32(img1_polar), np.float32(img2_polar))
        rotation, scale = ImageProcess.__scale_rotation(log_polar_cy, rows, cols)

        print_result['scale'] = scale
        print_result['rotation'] = rotation
        centre = (cols // 2, rows // 2)
        transformation_matrix = cv2.getRotationMatrix2D(centre, rotation, scale)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP | cv2.BORDER_REPLICATE
        (img2_y, img2_x, _) = img2.shape
        result_image = cv2.warpAffine(img2, transformation_matrix, dsize=(img2_x, img2_y), flags=flags)

        return result_image, print_result

    @staticmethod
    def shift_stabilization(img1, img2, print_result, crop_sizes):
        """
        Perform shift stabilization on two images using phase correlation with hanning window

    :param img1                     source image
        :param img2:                target image
        :param rows:                rows of result image
        :param cols:                columns of result image
        :param print_result:        gathered information during stabilization
        :return:
            result_image:   stabilized (shifted) image
            print_result:   collected information during shift stabilization
        """
        img1_gray = ImageProcess.crop_image(ImageProcess.to_gray(img1), crop_sizes)
        img2_gray = ImageProcess.crop_image(ImageProcess.to_gray(img2), crop_sizes)

        (rows, cols) = img1_gray.shape

        hanning = cv2.createHanningWindow((cols, rows), cv2.CV_32F)

        (cx, cy), _ = cv2.phaseCorrelate(np.float32(img1_gray), np.float32(img2_gray), hanning)
        (cx, cy) = (round(cx, 2), round(cy, 2))
        M = np.float32([[1, 0, cx], [0, 1, cy]])
        print_result['x'] = cx
        print_result['y'] = cy
        t_form = transform.EuclideanTransform(translation=(cx, cy))
        result_image = transform.warp(img2, t_form, order=1)
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

        # scale = math.exp(math.log(rows * 1.1 / 2.0) / max(rows, cols))
        # scale = 1.0 / math.pow(scale, cy)
        pcorr_shape = ImageProcess.__get_pcorr_shape((rows, cols))
        log_base = ImageProcess.__get_log_base((rows, cols), pcorr_shape[1])
        scale = log_base ** cy
        scale = 1.0 / scale

        return rotation, scale

    @staticmethod
    def __get_log_base(shape, new_r):
        old_r = shape[0] * 1.1
        # We are radius, so we divide the diameter by two.
        old_r /= 2.0
        # we have at most 'new_r' of space.
        log_base = np.exp(np.log(old_r) / new_r)
        return log_base

    @staticmethod
    def __get_pcorr_shape(shape):
        ret = (int(max(shape) * 1.0),) * 2
        return ret

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
    def crop_image(image, crop_sizes=(0, 0)):
        crop_x = int(crop_sizes[0] / 2)
        crop_y = int(crop_sizes[1] / 2)
        (x, y) = image.shape
        return image[crop_x:x - crop_x, crop_y:y - crop_y]

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
    def std(euclid_distances):
        return np.std(euclid_distances)

    @staticmethod
    def select_reference_points(image):
        points = []
        point_shape = 25
        alpha = 50
        beta = 1.5

        new_image = cv2.convertScaleAbs(image, alpha, beta)

        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                cv2.rectangle(new_image, (x - point_shape, y - point_shape), (x + point_shape, y + point_shape),
                              (255, 0, 0), 1)
                points.append((x, y))
                cv2.putText(new_image, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, cv2.LINE_AA)
                print(str(len(points)), ". point: ", (x, y))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', select_point)
        while True:
            cv2.imshow("image", new_image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break
            if len(points) == 5:
                break
        cv2.destroyAllWindows()

        gray = img_as_float(ImageProcess.to_gray(image))
        selected_points = {}
        for (x, y) in points:
            selected_points[(x, y)] = gray[y - point_shape:y + point_shape, x - point_shape:x + point_shape]

        print("Selected points:\t", str(selected_points.keys()))
        return selected_points

    @staticmethod
    def tracking_points(selected_points, image):
        tracked_points = []
        image = img_as_float(ImageProcess.to_gray(image))
        # for (x, y), value in selected_points.items():
        #     t_x, t_y = ImageProcess.find_point(image, x, y, value)
        #     tracked_points.append((t_x, t_y))
        # print("Tracked points:\t\t", tracked_points)

        for (x, y), template in selected_points.items():
            result = match_template(image, template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            tracked_point = ij[::-1]
            tracked_points.append(tracked_point)

        return tracked_points

    @staticmethod
    def find_point(image, x, y, value):
        (rows, cols) = image.shape
        neighbour = 15
        threshold = 0.0001
        for y_col in range(neighbour * -1, neighbour):
            tmp_y = y_col + y
            if cols <= tmp_y < 0:
                continue
            for x_row in range(neighbour * -1, neighbour):
                tmp_x = x_row + x
                if rows <= tmp_x < 0:
                    continue

                similar = abs(image[tmp_y, tmp_x] - value)
                if similar < threshold:
                    return tmp_x, tmp_y

        print("not found for: ", (x, y))
        return -1, -1

    @staticmethod
    def euclid_distance(selected_points, tracked_points):
        r = []
        for index in range(len(selected_points)):
            if tracked_points[index] != (-1, -1):
                r.append(distance.euclidean(list(selected_points.keys())[index], tracked_points[index]))
            else:
                r.append(-1)
        return r

