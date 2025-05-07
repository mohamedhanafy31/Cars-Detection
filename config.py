# config.py

DATA_DIR = 'data/cars_detection/Cars Detection/'
IMAGE_DIR = DATA_DIR + 'train/images/'
ANNOTATION_DIR = DATA_DIR + 'train/labels/'

NUM_CLASSES = 6  # background + 5 vehicle classes (Ambulance, Bus, Car, Motorcycle, Truck)
INPUT_SIZE = (224, 224)
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = 'cuda'  # or 'cpu'
