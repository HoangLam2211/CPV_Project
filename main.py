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
