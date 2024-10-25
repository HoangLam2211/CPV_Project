import timeit
import cv2
import numpy as np

# Lớp Matcher dùng để tìm và khớp các đặc trưng giữa các ảnh
class Matcher:
    def __init__(self):
        # Sử dụng thuật toán SIFT để phát hiện và mô tả đặc trưng
        self.sift = cv2.SIFT_create()
        # Thiết lập thuật toán FLANN để khớp đặc trưng giữa các ảnh
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Hàm để khớp các đặc trưng giữa hai ảnh
    def match(self, i1, i2):
        # Lấy đặc trưng SIFT cho ảnh thứ nhất và thứ hai
        image_set_1 = self.get_SIFT_features(i1)
        image_set_2 = self.get_SIFT_features(i2)

        # Khớp các đặc trưng giữa hai ảnh bằng FLANN
        matches = self.flann.knnMatch(image_set_2["des"], image_set_1["des"], k=2)

        # Danh sách các đặc trưng "tốt" sau khi lọc
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        # Nếu có đủ số lượng đặc trưng khớp tốt, tính ma trận Homography
        if len(good) > 4:
            points_current = image_set_2["kp"]  # Keypoints của ảnh hiện tại
            points_previous = image_set_1["kp"]  # Keypoints của ảnh trước đó

            # Lấy tọa độ các đặc trưng khớp
            matched_points_current = np.float32([points_current[i].pt for (__, i) in good])
            matched_points_prev = np.float32([points_previous[i].pt for (i, __) in good])

            # Tính toán ma trận Homography
            H, _ = cv2.findHomography(matched_points_current, matched_points_prev, cv2.RANSAC, 4)
            return H
        return None

    # Hàm lấy đặc trưng SIFT của một ảnh
    def get_SIFT_features(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang grayscale
        kp, des = self.sift.detectAndCompute(gray, None)  # Phát hiện và tính toán đặc trưng
        return {"kp": kp, "des": des}  # Trả về keypoints và descriptors


# Lớp Stitcher để ghép nối các ảnh thành một panorama
class Stitcher:
    def __init__(
        self,
        number_of_images,
        crop_x_min=None,
        crop_x_max=None,
        crop_y_min=None,
        crop_y_max=None,
    ):
        self.matcher_obj = Matcher()  # Đối tượng Matcher để khớp đặc trưng
        self.homography_cache = {}  # Bộ nhớ đệm cho ma trận Homography
        self.overlay_cache = {}  # Bộ nhớ đệm cho việc chồng ảnh
        self.count = number_of_images  # Số lượng ảnh cần ghép

        # Các thông số cắt ảnh cuối cùng
        self.crop_x_min = crop_x_min
        self.crop_x_max = crop_x_max
        self.crop_y_min = crop_y_min
        self.crop_y_max = crop_y_max

    # Hàm thực hiện ghép nối các ảnh
    def stitch(self, images=[]):
        self.images = images  # Danh sách ảnh
        self.prepare_lists()  # Chuẩn bị các danh sách ảnh

        # Bắt đầu đo thời gian ghép nối
        start = timeit.default_timer()
        self.left_shift()  # Ghép từ giữa sang trái
        self.right_shift()  # Ghép từ giữa sang phải
        stop = timeit.default_timer()
        duration = stop - start
        print("stitching took %.2f seconds." % duration)  # In ra thời gian thực hiện

        # Trả về ảnh kết quả đã cắt nếu có thông số crop
        if self.crop_x_min and self.crop_x_max and self.crop_y_min and self.crop_y_max:
            return self.result[
                self.crop_y_min : self.crop_y_max, self.crop_x_min : self.crop_x_max
            ]
        else:
            return self.result

    # Hàm chuẩn bị các danh sách ảnh để ghép
    def prepare_lists(self):
        # Danh sách ảnh trái và phải
        self.left_list = []
        self.right_list = []

        self.center_index = int(self.count / 2)  # Xác định ảnh ở giữa
        self.result = self.images[self.center_index]  # Bắt đầu với ảnh giữa

        # Chia các ảnh thành bên trái và bên phải so với ảnh giữa
        for i in range(self.count):
            if i <= self.center_index:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    # Hàm tính toán ma trận Homography giữa hai ảnh
    def get_homography(self, image_1, image_1_key, image_2, image_2_key, direction):
        cache_key = "_".join([image_1_key, image_2_key, direction])  # Tạo khóa bộ nhớ đệm
        homography = self.homography_cache.get(cache_key, None)
        if homography is None:
            homography = self.matcher_obj.match(image_1, image_2)  # Tính toán Homography
            self.homography_cache[cache_key] = homography  # Lưu vào bộ nhớ đệm
        return homography