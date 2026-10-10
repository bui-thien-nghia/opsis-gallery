from ultralytics import YOLOE

common_nouns = [
    # People and Relationships
    "person", "man", "woman", "child", "baby", "boy", "girl", "friend", "neighbor", "guest",
    "teacher", "student", "doctor", "nurse", "patient", "worker", "farmer", "lawyer", "driver", "actor",
    "parent", "father", "mother", "brother", "sister", "son", "daughter", "family", "partner", "boss",
    
    # Home and Furniture
    "house", "home", "room", "kitchen", "bedroom", "bathroom", "livingroom", "garage", "yard", "garden",
    "bed", "pillow", "blanket", "sheet", "closet", "drawer", "wardrobe", "mirror", "desk", "table", 
    "chair", "sofa", "couch", "bench", "stool", "shelf", "bookcase", "door", "window", "wall", 
    "floor", "ceiling", "roof", "gate", "fence", "carpet", "rug", "curtain", "lamp", "bulb",

    # Kitchen, Dining and Food
    "plate", "bowl", "cup", "mug", "glass", "bottle", "jar", "spoon", "fork", "knife", 
    "pot", "pan", "lid", "kettle", "toaster", "blender", "oven", "stove", "microwave", "fridge", 
    "sink", "faucet", "counter", "pantry", "basket", "bin", "trashcan", "sponge", "towel", "napkin",
    "water", "milk", "juice", "coffee", "tea", "bread", "butter", "cheese", "egg", "meat", 
    "chicken", "fish", "rice", "pasta", "flour", "sugar", "salt", "pepper", "soup", "salad",

    # Clothing and Accessories
    "shirt", "tshirt", "sweater", "jacket", "coat", "hoodie", "suit", "pants", "jeans", "shorts", 
    "skirt", "dress", "pajamas", "socks", "shoe", "boot", "sneaker", "sandal", "slipper", "hat", 
    "cap", "glove", "belt", "tie", "bag", "backpack", "purse", "wallet", "suitcase", "umbrella",

    # Technology and Office
    "phone", "smartphone", "charger", "cable", "headphone", "speaker", "computer", "laptop", "monitor", "screen", 
    "keyboard", "mouse", "printer", "tablet", "camera", "lens", "battery", "router", "television", "tv", 
    "remote", "clock", "watch", "calculator", "alarm", "book", "notebook", "pen", "pencil", "marker", 
    "eraser", "ruler", "paper", "sheet", "envelope", "letter", "card", "stamp", "tape", "glue",

    # Vehicles and Places
    "car", "truck", "van", "bus", "taxi", "bicycle", "bike", "motorcycle", "scooter", "train", 
    "airplane", "plane", "boat", "ship", "road", "street", "highway", "sidewalk", "bridge", "station", 
    "airport", "ticket", "passport", "luggage", "map", "building", "office", "factory", "school", "college", 
    "hospital", "clinic", "pharmacy", "store", "shop", "market", "supermarket", "mall", "bank", "hotel", 
    "restaurant", "cafe", "theater", "cinema", "museum", "park", "playground", "church", "farm", "barn"
]

# Prompt-free mode
# model = YOLOE("yoloe-26n-seg.pt")

# Custom class mode
model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(common_nouns)
model.save_prompt_embeddings("common_nouns.npz")

model = YOLOE("yoloe-26n-seg.pt")
model.load_prompt_embeddings("common_nouns.npz")

model.export(
    format="onnx",
    simplify=True,
    quantize=8,
    opset=17,
    imgsz=320,
    dynamic=False
)

# if __name__ == "__main__":
    # model.predict(
    #     r"..\..\images\140A3607.JPG",
    #     imgsz=320,
    #     save=True,
    #     conf=0.5,
    #     max_det=100
    # )
