"""
Create prompt to get pure backgrounds
"""
import json

templates = [
    "A real photo of {obj}",
]
classnames = [
    # indoor objects
    "empty living room", "empty kitchen",
    # vehicle
    "blue sky", "empty city street, color", "empty city road, color", "empty lake", "empty sea", "railway without train", "empty railway, color",
    # animal
    "trees", "forest", "empty street, colored", "farms", "nature", "empty farm", "stable"
]

to_save = {"background": {
    class_: [] for class_ in classnames
}}
for class_ in classnames:
    for temp in templates:
        print(temp.format(obj=class_))
    to_save["background"][class_] = [
        temp.format(obj=class_)
        for temp in templates
    ]
with open("background_templates.json", "w") as f:
    json.dump(to_save, f)