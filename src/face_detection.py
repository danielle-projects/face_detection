import cv2
from src.constants import FaceDetectionConstants


class FaceDetection:
    def __init__(self, img_path):
        self.face_cascade = cv2.CascadeClassifier(FaceDetectionConstants.CASCADE_CLASSIFIER_SRC)
        self.origin_face_img = cv2.imread(img_path)
        self.gray_img = None
        self.rectangle_faces_img = None
        self.resized_rectangle_faces_img = None

    def process_detect_faces(self):
        self.convert_img_to_gray()
        self.extract_faces_from_image()
        self.resize_img()
        self.show_img()

    def convert_img_to_gray(self):
        gray_img = cv2.cvtColor(self.origin_face_img, cv2.COLOR_BGR2GRAY)
        self.gray_img = gray_img

    def extract_faces_from_image(self):
        faces = self.face_cascade.detectMultiScale(
            self.gray_img,
            scaleFactor=FaceDetectionConstants.SCALE_FACTOR,
            minNeighbors=FaceDetectionConstants.MIN_NEIGHBORS,
            minSize=FaceDetectionConstants.MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print("Found {0} faces!".format(len(faces)))
        for x, y, w, h in faces:
            self.rectangle_faces_img = cv2.rectangle(self.gray_img, (x, y),
                                                     (x + w, y + h),
                                                     color=FaceDetectionConstants.RECTANGLE_COLOR,
                                                     thickness=FaceDetectionConstants.RECTANGLE_THICKNESS)

    def resize_img(self):
        img_shape = self.rectangle_faces_img.shape
        self.resized_rectangle_faces_img = cv2.resize(src=self.rectangle_faces_img,
                                                      dsize=(int(img_shape[1] / 3),
                                                             int(img_shape[0] / 3)))

    def show_img(self):
        cv2.imshow(FaceDetectionConstants.WINDOW_NAME, self.resized_rectangle_faces_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
