from src.face_detection import FaceDetection


def main():
    FaceDetection(img_path="src/statics/fake_ai_faces.0.png").process_detect_faces()



# def main():
#     face_cascade = cv2.CascadeClassifier("src/statics/haarcascade_frontalface_default.xml")
#     face_img = cv2.imread("src/statics/news.jpg")
#     gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray_img,
#                                           scaleFactor=1.3,
#                                           minNeighbors=5,
#                                           minSize=(30, 30),
#                                           flags=cv2.CASCADE_SCALE_IMAGE)
#     for x, y, w, h in faces:
#         face_img = cv2.rectangle(face_img, (x, y),
#                                  (x + w, y + h),
#                                  (0, 255, 0),
#                                  2)
#
#     resized = cv2.resize(face_img,
#                          (int(face_img.shape[1] / 3),
#                           int(face_img.shape[0] / 3)))
#
#     cv2.imshow("Gray", resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

