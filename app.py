import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import queue
from pathlib import Path

# import tensorflow as tf
# import tensorflow_hub as hub
import time, sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import numpy as np
import time
import sys
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import streamlit_webrtc as webrtc
import logging
from typing import List, NamedTuple
from download import download_file

## 參考至 https://github.com/zhoroh/ObjectDetection ##


def object_detection_video():
    # object_detection_video.has_beenCalled = True
    # pass
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = (
        r"C:\Users\user\Desktop\test\ObjectDetection\config_n_weights\yolov3.cfg"
    )
    weights_path = (
        r"C:\Users\user\Desktop\test\ObjectDetection\config_n_weights\yolov3.weights"
    )
    font_scale = 1
    thickness = 1
    # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
    # f = urllib.request.urlopen(url)
    # labels = [line.decode("utf-8").strip() for line in f]
    f = open(r"C:\Users\user\Desktop\test\ObjectDetection\labels\coconames.txt", "r")
    lines = f.readlines()
    labels = [line.strip() for line in lines]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    st.title("影片偵測")
    st.subheader(
        """
    下方可以輸入影片來提供食材進行辨識
    """
    )
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mpeg", "mov"])
    if uploaded_video != None:
        vid = uploaded_video.name
        with open(vid, mode="wb") as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(vid, "rb")
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        # video_file = 'street.mp4'
        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]
        # out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc#(*'avc3'), fps, insize)

        fourcc = cv2.VideoWriter_fourcc(*"mpv4")
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0
        while True:
            _, image = cap.read()
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    image, 1 / 255.0, (416, 416), swapRB=True, crop=False
                )
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count += 1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(
                    boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD
                )

                font_scale = 0.6
                thickness = 1

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(
                            image,
                            (x, y),
                            (x + w, y + h),
                            color=color,
                            thickness=thickness,
                        )
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            thickness=thickness,
                        )[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = (
                            (text_offset_x, text_offset_y),
                            (
                                text_offset_x + text_width + 2,
                                text_offset_y - text_height,
                            ),
                        )
                        overlay = image.copy()
                        cv2.rectangle(
                            overlay,
                            box_coords[0],
                            box_coords[1],
                            color=color,
                            thickness=cv2.FILLED,
                        )
                        # add opacity (transparency to the box)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(
                            image,
                            text,
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            color=(0, 0, 0),
                            thickness=thickness,
                        )

                out.write(image)
                cv2.imshow("image", image)

                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break

        # return "detected_video.mp4"

        cap.release()
        cv2.destroyAllWindows()


def object_detection_image():
    st.title("圖片偵測")
    st.subheader(
        """
    下方可以輸入照片來提供食材進行辨識
    """
    )
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption="Uploaded Image")
        my_bar = st.progress(0)
        confThreshold = st.slider("Confidence", 0, 100, 50)
        nmsThreshold = st.slider("Threshold", 0, 100, 20)
        # classNames = []
        whT = 320
        # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
        # f = urllib.request.urlopen(url)
        # classNames = [line.decode("utf-8").strip() for line in f]
        f = open(
            r"C:\Users\user\Desktop\test\ObjectDetection\labels\coconames.txt", "r"
        )
        lines = f.readlines()
        classNames = [line.strip() for line in lines]
        config_path = (
            r"C:\Users\user\Desktop\test\ObjectDetection\config_n_weights\yolov3.cfg"
        )
        weights_path = r"C:\Users\user\Desktop\test\ObjectDetection\config_n_weights\yolov3.weights"
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs, img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold / 100):
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                        bbox.append([x, y, w, h])
                        classIds.append(classId)
                        confs.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(
                bbox, confs, confThreshold / 100, nmsThreshold / 100
            )
            obj_list = []
            confi_list = []
            # drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (240, 54, 230), 2)
                # print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())

                confi_list.append(int(confs[i] * 100))
                cv2.putText(
                    img2,
                    f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (240, 0, 240),
                    2,
                )
            df = pd.DataFrame(
                list(zip(obj_list, confi_list)), columns=["Object Name", "Confidence"]
            )
            if st.checkbox("Show Object's list"):
                st.write(df)
            if st.checkbox("Show Confidence bar chart"):
                st.subheader("Bar chart for confidence levels")

                st.bar_chart(df["Confidence"])

        blob = cv2.dnn.blobFromImage(
            img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False
        )
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img2)

        st.image(img2, caption="Proccesed Image.")

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        my_bar.progress(100)


def object_detection_webcam():
    st.title("視訊偵測")
    st.subheader(
        """
        請先點選左下方的Start按鈕來啟動Webcam，系統會即時偵測您的影像中是否有特定物件存在。
        """
    )


def main():
    new_title = '<p style="font-size: 42px;">新家庭主夫</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown(
        """
    現在，在家中輕鬆地做出一道讓你和家人嘴巴滿足的料理！使用我們的全新App，結合YOLO物體偵測技術和AI智能分析，讓你輕鬆拍下一張照片或影片，自動識別食材並為您推薦適合的食譜！現在就來試試看吧，讓我們為您的烹飪體驗加點色彩！
    """
    )
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("MODE", ("關於", "物件偵測（圖片）", "物件偵測（影片）", "物件偵測（視訊）"))
    if choice == "物件偵測（圖片）":
        # st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        # st.title('Object Detection')
        object_detection_image()
    elif choice == "物件偵測（影片）":
        read_me_0.empty()
        read_me.empty()
        object_detection_video()
        try:
            clip = moviepy.VideoFileClip("detected_video.mp4")
            clip.write_videofile("detected_video.mp4")
            st_video = open("detected_video.mp4", "rb")
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video")
        except OSError:
            """"""
    elif choice == "關於":
        st.write(
            "我們使用[iCook愛料理](https://icook.tw/)—台灣最大料理生活平台，做為資料源，總共包含80000種不同的食譜，可以用不同方式上傳各種食材進行辨認，讓我們幫你推薦最適合的美食！"
        )

        st.write("成員：曾子朋、陳柏維、陳劭晏、葉秉鈞")

    elif choice == "物件偵測（視訊）":
        read_me_0.empty()
        read_me.empty()
        object_detection_webcam()

        # 參考至 https://github.com/whitphx/streamlit-webrtc/blob/main/pages/1_object_detection.py
        HERE = Path(__file__).parent
        ROOT = HERE.parent
        logger = logging.getLogger(__name__)

        MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
        MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
        PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
        PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

        CLASSES = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        class Detection(NamedTuple):
            class_id: int
            label: str
            score: float
            box: np.ndarray

        @st.cache_resource  # type: ignore
        def generate_label_colors():
            return np.random.uniform(0, 255, size=(len(CLASSES), 3))

        COLORS = generate_label_colors()

        download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

        # Session-specific caching
        cache_key = "object_detection_dnn"
        if cache_key in st.session_state:
            net = st.session_state[cache_key]
        else:
            net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            st.session_state[cache_key] = net

        score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
        result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            # Run inference
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            net.setInput(blob)
            output = net.forward()

            h, w = image.shape[:2]

            # Convert the output array into a structured form.
            output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
            output = output[output[:, 2] >= score_threshold]
            detections = [
                Detection(
                    class_id=int(detection[1]),
                    label=CLASSES[int(detection[1])],
                    score=float(detection[2]),
                    box=(detection[3:7] * np.array([w, h, w, h])),
                )
                for detection in output
            ]

            # Render bounding boxes and captions
            for detection in detections:
                caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
                color = COLORS[detection.class_id]
                xmin, ymin, xmax, ymax = detection.box.astype("int")

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(
                    image,
                    caption,
                    (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            result_queue.put(detections)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                while True:
                    result = result_queue.get()
                    labels_placeholder.table(result)


if __name__ == "__main__":
    main()
