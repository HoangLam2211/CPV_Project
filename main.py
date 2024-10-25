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
        print("Hoan thanh %.2f giay." % duration)  # In ra thời gian thực hiện

        # Trả về ảnh kết quả đã cắt nếu có thông số crop
        if self.crop_x_min and self.crop_x_max and self.crop_y_min and self.crop_y_max:
            return self.result
            [
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

    # Hàm ghép ảnh từ giữa sang trái
    def left_shift(self):
        a = self.left_list[0]  # Bắt đầu với ảnh đầu tiên bên trái

        # Ghép từng ảnh trong danh sách bên trái
        for i, image in enumerate(self.left_list[1:]):
            H = self.get_homography(a, str(i), image, str(i + 1), "left")  # Tính Homography

            # Tính toán nghịch đảo ma trận Homography
            XH = np.linalg.inv(H)

            # Xác định kích thước của ảnh biến đổi
            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]

            # Điều chỉnh các giá trị bù trừ
            f1 = np.dot(XH, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            XH[0][-1] += abs(f1[0])
            XH[1][-1] += abs(f1[1])

            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))

            # Biến đổi phối cảnh của ảnh
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            tmp = cv2.warpPerspective(a, XH, dsize, borderMode=cv2.BORDER_TRANSPARENT)

            # Chèn ảnh hiện tại vào ảnh ghép
            tmp[offsety : image.shape[0] + offsety, offsetx : image.shape[1] + offsetx] = image

            a = tmp  # Cập nhật ảnh ghép

        self.result = tmp  # Lưu ảnh kết quả

    # Hàm ghép ảnh từ giữa sang phải
    def right_shift(self):
        for i, imageRight in enumerate(self.right_list):
            imageLeft = self.result  # Ảnh hiện tại bên trái

            H = self.get_homography(imageLeft, str(i), imageRight, str(i + 1), "right")  # Tính Homography

            # Biến đổi phối cảnh cho ảnh phải
            result = cv2.warpPerspective(
                imageRight,
                H,
                (imageLeft.shape[1] + imageRight.shape[1], imageLeft.shape[0]),
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            # Tạo mặt nạ cho ảnh trái
            mask = np.zeros((result.shape[0], result.shape[1], 3), dtype="uint8")
            mask[0 : imageLeft.shape[0], 0 : imageLeft.shape[1]] = imageLeft

            # Ghép hai ảnh
            self.result = self.blend_images(mask, result, str(i))

    # Hàm để chồng ảnh, giữ phần ảnh bên phải
    def blend_images(self, background, foreground, i):
        only_right = self.overlay_cache.get(i, None)
        if only_right is None:
            only_right = np.nonzero(
                (np.sum(foreground, 2) != 0) * (np.sum(background, 2) == 0)
            )
            self.overlay_cache[i] = only_right

        # Chỉ giữ phần ảnh bên phải
        background[only_right] = foreground[only_right]
        return background

def shanghai():
    
    FRAME_WIDTH = 768  # Chiều rộng ảnh
    FRAME_HEIGHT = 432  # Chiều cao ảnh

    # Danh sách các file ảnh
    shanghai_files = [
        "images/shanghai-01.png",
        "images/shanghai-02.png",
        "images/shanghai-03.png",
        "images/shanghai-04.png",
        "images/shanghai-05.png",
    ]

    # Đọc và thay đổi kích thước ảnh
    shanghai = [cv2.resize(cv2.imread(f), (FRAME_WIDTH, FRAME_HEIGHT)) for f in shanghai_files]

    # Thông số để cắt ảnh
    crop_x_min = 30 # Tọa độ x bắt đầu của vùng cắt.
    crop_x_max = 1764 # Tọa độ x kết thúc của vùng cắt.
    crop_y_min = 37 # Tọa độ y bắt đầu của vùng cắt.
    crop_y_max = 471 # Tọa độ y kết thúc của vùng cắt.

    # Khởi tạo đối tượng Stitcher và thực hiện ghép ảnh
    s = Stitcher(
        len(shanghai_files),
        crop_x_min=crop_x_min,
        crop_x_max=crop_x_max,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )

    panorama = s.stitch(shanghai)  # Ghép ảnh

    cv2.imwrite("panorama1.png", panorama)  # Lưu ảnh kết quả
    cv2.imshow('ddd', panorama)  # Hiển thị ảnh kết quả
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị

def building():
    
    FRAME_WIDTH = 1000  # Chiều rộng ảnh
    FRAME_HEIGHT = 500  # Chiều cao ảnh

    # Danh sách các file ảnh
    building_files = [
        "input_image\\building\\building1.jpg",
        "input_image\\building\\building2.jpg",
        "input_image\\building\\building3.jpg",
        "input_image\\building\\building4.jpg",
        "input_image\\building\\building5.jpg",
    ]

    # Đọc và thay đổi kích thước ảnh
    building = [cv2.resize(cv2.imread(f), (FRAME_WIDTH, FRAME_HEIGHT)) for f in building_files]

    # Thông số để cắt ảnh
    crop_x_min = 30 # Tọa độ x bắt đầu của vùng cắt.
    crop_x_max = 2500 # Tọa độ x kết thúc của vùng cắt.
    crop_y_min = 37 # Tọa độ y bắt đầu của vùng cắt.
    crop_y_max = 1000 # Tọa độ y kết thúc của vùng cắt.

    # Khởi tạo đối tượng Stitcher và thực hiện ghép ảnh
    s = Stitcher(
        len(building_files),
        crop_x_min=crop_x_min,
        crop_x_max=crop_x_max,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )

    panorama = s.stitch(building)  # Ghép ảnh

    cv2.imwrite("panorama2.png", panorama)  # Lưu ảnh kết quả
    cv2.imshow('ddd', panorama)  # Hiển thị ảnh kết quả
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị

def city():
    
    FRAME_WIDTH = 1000  # Chiều rộng ảnh
    FRAME_HEIGHT = 500  # Chiều cao ảnh

    # Danh sách các file ảnh
    city_files = [
        "input_image\\city\\002.jpg",
        "input_image\\city\\003.jpg",
        "input_image\\city\\004.jpg",
        "input_image\\city\\005.jpg",
        "input_image\\city\\006.jpg",
        "input_image\\city\\007.jpg",
        "input_image\\city\\008.jpg",
    ]

    # Đọc và thay đổi kích thước ảnh
    city = [cv2.resize(cv2.imread(f), (FRAME_WIDTH, FRAME_HEIGHT)) for f in city_files]

    # Thông số để cắt ảnh
    crop_x_min = 50 # Tọa độ x bắt đầu của vùng cắt.
    crop_x_max = 5000 # Tọa độ x kết thúc của vùng cắt.
    crop_y_min = 50 # Tọa độ y bắt đầu của vùng cắt.
    crop_y_max = 1200 # Tọa độ y kết thúc của vùng cắt.

    # Khởi tạo đối tượng Stitcher và thực hiện ghép ảnh
    s = Stitcher(
        len(city_files),
        crop_x_min=crop_x_min,
        crop_x_max=crop_x_max,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )

    panorama = s.stitch(city)  # Ghép ảnh

    cv2.imwrite("panorama3.png", panorama)  # Lưu ảnh kết quả
    cv2.imshow('ddd', panorama)  # Hiển thị ảnh kết quả
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị

def street():
    
    # Inside the mountain() function
    street_files = [
        "input_image\\street\\001.jpg",
        "input_image\\street\\002.jpg",
        "input_image\\street\\003.jpg",
        "input_image\\street\\004.jpg",
        "input_image\\street\\005.jpg",
    ]

    # Set a consistent width and height
    FRAME_WIDTH = 900
    FRAME_HEIGHT = 600

    # Resize all images to the same size
    street = [cv2.resize(cv2.imread(f), (FRAME_WIDTH, FRAME_HEIGHT)) for f in street_files]

    # Thông số để cắt ảnh
    crop_x_min = 50 # Tọa độ x bắt đầu của vùng cắt.
    crop_x_max = 5000 # Tọa độ x kết thúc của vùng cắt.
    crop_y_min = 50 # Tọa độ y bắt đầu của vùng cắt.
    crop_y_max = 1200 # Tọa độ y kết thúc của vùng cắt.

    # Khởi tạo đối tượng Stitcher và thực hiện ghép ảnh
    s = Stitcher(
        len(street_files),
        crop_x_min=crop_x_min,
        crop_x_max=crop_x_max,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )

    panorama = s.stitch(street)  # Ghép ảnh

    cv2.imwrite("panorama4.png", panorama)  # Lưu ảnh kết quả
    cv2.imshow('ddd', panorama)  # Hiển thị ảnh kết quả
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị



if __name__ == "__main__":
    shanghai()
    building()
    city()
    street()
