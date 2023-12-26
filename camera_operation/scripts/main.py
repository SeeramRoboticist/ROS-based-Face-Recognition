#!/usr/bin/env python

from pathlib import Path

import pickle

import face_recognition

from collections import Counter

import cv2

import rospy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from std_msgs.msg import String, Bool

DEFAULT_ENCODINGS_PATH = Path("/home/asimov/face_reg_ws/src/output/encodings.pkl")

class FaceRecognition:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)

        rospy.Subscriber("/face_control", Bool, self.control_cb, queue_size=1)

        self.face_result = rospy.Publisher("/face_result", String, queue_size=1)

        self.image_pub = rospy.Publisher("/front_cam", Image, queue_size=1)

        self.recognition_control = True

        # self.bounding_box = ()

    def control_cb(self, data):

        if data.data == True:
            self.recognition_control = True
        else:
            self.recognition_control = False

    def image_callback(self, camera_data):

        frame_src = self.bridge.imgmsg_to_cv2(camera_data, "bgr8")
        (h_src, w_src) = frame_src.shape[:2]

        crop_height = int(h_src / 2)
        frame_edited = frame_src[0:(crop_height - 1), 0:w_src]
        self.frame = frame_src[crop_height:h_src, 0:w_src]

        # test_img = cv2.flip(test_img, 1)

        # self.recognize_faces(camera_frame=frame)

        if self.recognition_control:

            # print("hello_world")

            rgb1 = cv2.resize(frame_edited, (int(frame_edited.shape[1] / 2), int(frame_edited.shape[0] / 2)))
            rgb = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
            self.r = frame_edited.shape[1] / float(rgb.shape[1])

            self.recognize_faces(camera_frame=rgb)

            # for ((top, right, bottom, left)) in zip(self.bounding_box):
            #     # rescale the face coordinates
            
            # top, right, bottom, left = self.bounding_box

            # top = int(top * r)
            # right = int(right * r)
            # bottom = int(bottom * r)
            # left = int(left * r)

            # cv2.rectangle(frame, (left, top), (right, bottom),
            #             (0, 255, 0), 2)
            
            # y = top - 15 if top - 15 > 15 else top + 15

            # if self.name == "Unknown":
            #     cv2.putText(frame, "Visitor", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            # else:
            #     cv2.putText(frame, self.name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                
            # frame1 = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            # self.image_pub.publish(frame1)


        # self.image_pub.publish(test_img)

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


        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):

            name = self._recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"

            print(name, bounding_box)

            # for ((top, right, bottom, left)) in zip(self.bounding_box):
                # rescale the face coordinates
            
            top, right, bottom, left = bounding_box

            top = int(top * self.r)
            right = int(right * self.r)
            bottom = int(bottom * self.r)
            left = int(left * self.r)

            cv2.rectangle(self.frame, (left, top), (right, bottom),
                        (0, 255, 0), 2)
            
            y = top - 15 if top - 15 > 15 else top + 15

            if name == "Unknown":
                cv2.putText(self.frame, "Visitor", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            else:
                cv2.putText(self.frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                
            frame1 = self.bridge.cv2_to_imgmsg(self.frame, 'bgr8')
            self.image_pub.publish(frame1)

            # return bounding_box

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
        





