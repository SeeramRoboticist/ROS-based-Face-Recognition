from pathlib import Path

import face_recognition

import pickle

from collections import Counter

from PIL import Image, ImageDraw

import cv2

import time

video_capture = cv2.VideoCapture(0)

DEFAULT_ENCODINGS_PATH = Path("/home/sriram54/face_reg/output/encodings.pkl")

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


def face_reg():

    global bounding_box
    bounding_box = ()

    while True:

        ret, frame = video_capture.read()


        cv2.imshow('lol', frame)

        recognize_faces(image_location=frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
    
    video_capture.release()
    cv2.destroyAllWindows()



def recognize_faces(
    image_location,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    global name, bounding_box
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        image_location, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        image_location, input_face_locations
    )

    # pillow_image = Image.fromarray(input_image)
    # draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"

        print(name)

        # _display_face(draw, bounding_box, name)

    # del draw
    # pillow_image.show()
            
    # return name


def _recognize_face(unknown_encoding, loaded_encodings):
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
    
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

def validate(model: str = "hog"):
    for filepath in Path("/home/sriram54/face_recognizer/training").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


# recognize_faces("/home/sriram54/Downloads/lol.jpeg")

# encode_known_faces()
            
# validate()
            
face_reg()