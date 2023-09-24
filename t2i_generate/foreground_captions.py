import json

templates = [
    "a photo of {obj}",
    "a realistic photo of {obj}",
    "a photo of {obj} in pure background",
    "{obj} in a white background",
    "{obj} without background",
    "{obj} isolated on white background",
]

classnames = [
    # 'a truck', 'a traffic light', 'a fire hydrant', 'a stop sign', 'a parking meter', 'a bench',
    # 'an elephant', 'a bear', 'a zebra', 'a giraffe', 'a backpack', 'an umbrella',
    # 'a handbag', 'a tie', 'a suitcase', 'a frisbee', 'a ski', 'a snowboard', 'a sports ball', 'a kite', 'a baseball bat',
    # 'a baseball glove', 'a skateboard', 'a surfboard', 'a tennis racket', 'a wine glass', 'a cup', 'a fork',
    # 'a knife', 'a spoon', 'a bowl', 'a banana', 'an apple', 'a sandwich', 'an orange', 'a broccoli', 'a carrot', 'a hot dog',
    # 'a pizza', 'a donut', 'a cake', 'a couch', 'a bed', 'a toilet',
    # 'a laptop', 'a computer mouse', 'an electronic remote', 'a keyboard', 'a cell phone', 'a microwave', 'an oven', 'a toaster', 'a sink', 'a refrigerator',
    # 'a book', 'a clock', 'a vase', 'a scissors', 'a teddy bear', 'a hair drier', 'a toothbrush',
    # above are 60 classes, used in COCO
    "a person", "a man", "a woman",
    "a bird", "a cat", "a cow", "a dog", "a horse", "a sheep", 
    "an airplane", 
    "a TV", "a monitor", "an old monitor", "a dining table", "a table", 
    "a bicycle", "a boat", "a bus", "a car", "a motorbike", "a train",
    "a bottle", "a chair", "a dining table", "a potted plant", "a sofa", "a tv monitor"
]
# ]

to_save = {"foreground": {
    class_: [] for class_ in classnames
}}
for class_ in classnames:
    for temp in templates:
        print(temp.format(obj=class_))
    to_save["foreground"][class_] = [
        temp.format(obj=class_)
        for temp in templates
    ]
with open("foreground_templates.json", "w") as f:
    json.dump(to_save, f)