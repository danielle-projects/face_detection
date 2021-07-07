import cv2
from src.constants import FaceDetectionConstants


class FaceDetection:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FaceDetectionConstants.CASCADE_CLASSIFIER_SRC)
        self.gray_img = None
        self.origin_img = None
        self.resized_faces_img = None
        self.rectangle_faces_img = None

    def process_detect_faces(self, img_path):
        self.origin_img = ImgProcess.read_img(img_path)
        self.gray_img = ImgProcess.convert_img_to_gray(self.origin_img)
        self.extract_faces_from_image(self.gray_img)
        self.resized_faces_img = ImgProcess.resize_img(self.rectangle_faces_img)
        ImgProcess.show_img(self.resized_faces_img)

    def extract_faces_from_image(self, gray_img):
        faces = self.face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=FaceDetectionConstants.SCALE_FACTOR,
            minNeighbors=FaceDetectionConstants.MIN_NEIGHBORS,
            minSize=FaceDetectionConstants.MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print("Found {0} faces!".format(len(faces)))
        for x, y, w, h in faces:
            self.rectangle_faces_img = cv2.rectangle(self.origin_img, (x, y),
                                                     (x + w, y + h),
                                                     color=FaceDetectionConstants.RECTANGLE_COLOR,
                                                     thickness=FaceDetectionConstants.RECTANGLE_THICKNESS)


class ImgProcess:

    @staticmethod
    def read_img(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def convert_img_to_gray(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img

    @staticmethod
    def resize_img(img):
        return cv2.resize(src=img,
                          dsize=(int(img.shape[1] / 3),
                                 int(img.shape[0] / 3)))

    @staticmethod
    def show_img(img):
        cv2.imshow(FaceDetectionConstants.WINDOW_NAME, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
