import numpy as np
import random

IMG_SIZE = 64
DIGIT_SIZE = 14
SEQ_LEN = 20


class DigitObject:
    def __init__(self, img, x, y, vx, vy, active=True):
        self.img = img
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.active = active

    def step(self, motion, collision_mode):
        if not self.active:
            return

        if motion == "vertical":
            self.vx = 0
        elif motion == "horizontal":
            self.vy = 0

        self.x += self.vx
        self.y += self.vy

        if self.x <= 0 or self.x + DIGIT_SIZE >= IMG_SIZE:
            if collision_mode == "bounce":
                self.vx *= -1
            else:
                self.vx = 0

        if self.y <= 0 or self.y + DIGIT_SIZE >= IMG_SIZE:
            if collision_mode == "bounce":
                self.vy *= -1
            else:
                self.vy = 0


def overlap(o1, o2):
    return not (
        o1.x + DIGIT_SIZE < o2.x or
        o1.x > o2.x + DIGIT_SIZE or
        o1.y + DIGIT_SIZE < o2.y or
        o1.y > o2.y + DIGIT_SIZE
    )


def handle_object_collision(o1, o2, collision_mode):
    if not (o1.active and o2.active):
        return

    if overlap(o1, o2):
        if collision_mode == "bounce":
            o1.vx, o2.vx = o2.vx, o1.vx
            o1.vy, o2.vy = o2.vy, o1.vy
        else:
            o1.vx = o1.vy = 0
            o2.vx = o2.vy = 0


def render(objects):
    frame = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for obj in objects:
        if not obj.active:
            continue
        h, w = obj.img.shape

        x0 = max(0, int(obj.x))
        y0 = max(0, int(obj.y))
        x1 = min(IMG_SIZE, x0 + w)
        y1 = min(IMG_SIZE, y0 + h)

        img_x1 = w - (x0 + w - x1)
        img_y1 = h - (y0 + h - y1)

        frame[y0:y1, x0:x1] = np.maximum(
            frame[y0:y1, x0:x1],
            obj.img[:img_y1, :img_x1]
        )
    return frame



def generate_sequence(mnist_by_digit, config):
    """
    mnist_by_digit: dict[int] -> list of digit images
    config: one of the configs above
    """

    objects = []

    for i in range(2):  # max 2 digits
        digit = random.choice(config["digit_range"])
        img = random.choice(mnist_by_digit[digit])

        x, y = random.randint(0, 48), random.randint(0, 48)
        vx, vy = random.choice([-2, 2]), random.choice([-2, 2])
        active = i < config["num_digits"]

        objects.append(DigitObject(img, x, y, vx, vy, active))

    anomaly_frame = SEQ_LEN // 2
    frames = []

    for t in range(SEQ_LEN):

        if config["anomaly"] and t == anomaly_frame:
            if config["anomaly"] == "stick":
                config["collision_mode"] = "stick"
            elif config["anomaly"] == "disappear":
                objects[0].active = False
            elif config["anomaly"] == "appear":
                objects[1].active = True

        for obj in objects:
            obj.step(config["motion"], config["collision_mode"])

        handle_object_collision(objects[0], objects[1], config["collision_mode"])
        frames.append(render(objects))

    return np.stack(frames)

from load_mnist import load_mnist_by_digit
from generator import generate_sequence
from visualize import save_gif
from config import ID_APPEAR

mnist_by_digit = load_mnist_by_digit()
seq = generate_sequence(mnist_by_digit, ID_APPEAR)

save_gif(seq, "ID_APPEAR_example.gif")
