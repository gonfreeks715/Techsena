from roboflow import Roboflow

rf = Roboflow(api_key="TVhJOzliR15d4kqiQS2M")

project = rf.workspace("gons-workspace").project("xray-weapon-detection")

dataset = project.version(1).download("yolov8", location="../datasets/xray")