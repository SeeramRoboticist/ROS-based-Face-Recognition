#!/usr/bin/env python

from pathlib import Path

import pickle

import face_recognition

from collections import Counter

import cv2

import rospy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from std_msgs.msg import String

DEFAULT_ENCODINGS_PATH = Path("/home/sriram54/face_recognizer/src/output/encodings.pkl")

class FaceRecognition:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/webcam/image_raw", Image, self.image_callback, queue_size=1)

        self.face_result = rospy.Publisher("/face_result", String, queue_size=1)

        self.image_pub = rospy.Publisher("/front_cam", Image, queue_size=1)


    def image_callback(self, camera_data):

        frame_src = self.bridge.imgmsg_to_cv2(camera_data, "bgr8")
        (h_src, w_src) = frame_src.shape[:2]

        crop_height = int(h_src / 2)
        frame_edited = frame_src[0:(crop_height - 1), 0:w_src]
        test_img = frame_src[crop_height:h_src, 0:w_src]


        self.recognize_faces(camera_frame=test_img)
        self.image_pub.publish(test_img)

    def recognize_faces(self, camera_frame, model: str = "hog",
                        encodings_location: Path = DEFAULT_ENCODINGS_PATH,) -> None:
        
        with encodings_location.open(mode="rb") as f:

            loaded_encodings = pickle.load(f)

        # input_image = face_recognition.load_image_file(camera_frame)

        input_face_locations = face_recognition.face_locations(
            camera_frame, model=model
        )
        input_face_encodings = face_recognition.face_encodings(
            camera_frame, input_face_locations
        )

        # pillow_image = Image.fromarray(input_image)
        # draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):

            self.name = self._recognize_face(unknown_encoding, loaded_encodings)
            if not self.name:
                self.name = "Unknown"

            print(self.name)

    def _recognize_face(self, unknown_encoding, loaded_encodings):

        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            if match
        )
        if votes:
            return votes.most_common(1)[0][0]
        

if __name__ == "__main__":

    rospy.init_node("face_reg", anonymous=True)
    obj = FaceRecognition()
    rospy.spin()
        





