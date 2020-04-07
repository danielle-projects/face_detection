from src.face_detection import FaceDetection


def main():
    FaceDetection().process_detect_faces(img_path="src/statics/fake_ai_faces.0.png")


if __name__ == '__main__':
    main()

