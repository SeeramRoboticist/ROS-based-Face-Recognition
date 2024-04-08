#!/usr/bin/env python

from pathlib import Path

import face_recognition

import pickle

import rospy

from std_msgs.msg import Bool, String

from time import sleep

DEFAULT_ENCODINGS_PATH = Path("/home/sriram54/face_reg/output/encodings.pkl")
process_start = False


def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    for filepath in Path("/home/sriram54/face_reg/training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
    print("completed")

    return "done"

def train_control_cb(data):
    global process_start
    if data.data == True:
        process_start = True
    else:
        process_start = False

if __name__ == "__main__":

    rospy.init_node("train_face", anonymous=True)
    rospy.Subscriber("train_control", Bool, train_control_cb, queue_size=1)
    pub = rospy.Publisher('/train_result', String, queue_size=1)
    while not rospy.is_shutdown():
        sleep(1)
        if process_start:
            res = encode_known_faces()
            process_start = False
            if res == "done":
                pub.publish("done")
            else:
                pub.publish("not_done")
    