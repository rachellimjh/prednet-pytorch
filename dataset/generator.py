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

        # Restrict motion
        if motion == "vertical":
            self.vx = 0
        elif motion == "horizontal":
            self.vy = 0

        # Move
        self.x += self.vx
        self.y += self.vy

        # ---- Wall collision ----
        # Left / right wall
        if self.x <= 0:
            self.x = 0
            if collision_mode == "bounce":
                self.vx *= -1
            else:  # slide
                self.vx = 0

        elif self.x + DIGIT_SIZE >= IMG_SIZE:
            self.x = IMG_SIZE - DIGIT_SIZE
            if collision_mode == "bounce":
                self.vx *= -1
            else:
                self.vx = 0

        # Top / bottom wall
        if self.y <= 0:
            self.y = 0
            if collision_mode == "bounce":
                self.vy *= -1
            else:
                self.vy = 0

        elif self.y + DIGIT_SIZE >= IMG_SIZE:
            self.y = IMG_SIZE - DIGIT_SIZE
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


def handle_object_collision(objects, collision_mode):
    """Handle collisions between all active objects"""
    active_objects = [obj for obj in objects if obj.active]
    
    if collision_mode == "slide":
        return
    
    # Check all pairs of active objects
    for i in range(len(active_objects)):
        for j in range(i + 1, len(active_objects)):
            o1, o2 = active_objects[i], active_objects[j]
            
            if overlap(o1, o2):
                # Bounce: swap velocities
                o1.vx, o2.vx = o2.vx, o1.vx
                o1.vy, o2.vy = o2.vy, o1.vy

                # Separate objects to avoid jitter
                o1.x += o1.vx
                o1.y += o1.vy
                o2.x += o2.vx
                o2.y += o2.vy



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
    objects = []
    
    # ---- Determine number of objects needed ----
    max_objects = 2  # Default
    num_active = config["num_digits"]
    
    if config.get("anomaly") == "appear":
        max_objects = 3  # Need 3 total: 2 active + 1 inactive
        num_active = 2   # Start with 2 active
    elif config.get("anomaly") == "disappear":
        max_objects = 2  # Need 2 total
        num_active = 2   # Start with 2 active, will become 1

    # ---- Create objects ----
    for i in range(max_objects):
        digit = random.choice(config["digit_range"])
        img = random.choice(mnist_by_digit[digit])

        x = random.randint(0, IMG_SIZE - DIGIT_SIZE)
        y = random.randint(0, IMG_SIZE - DIGIT_SIZE)
        vx = random.choice([-2, 2])
        vy = random.choice([-2, 2])

        active = i < num_active
        objects.append(DigitObject(img, x, y, vx, vy, active))

    # ---- Fix: Position objects to guarantee collision for slide anomaly ----
    if config.get("anomaly") == "slide" and config["num_digits"] == 2:
        # Keep random velocities, but position objects to collide
        # Place them on opposite sides, moving toward center
        objects[0].x = 5
        objects[0].y = IMG_SIZE // 2 - DIGIT_SIZE // 2
        objects[0].vx = 2
        objects[0].vy = random.choice([-2, 2])  # Keep vertical motion
        
        objects[1].x = IMG_SIZE - DIGIT_SIZE - 5
        objects[1].y = IMG_SIZE // 2 - DIGIT_SIZE // 2
        objects[1].vx = -2
        objects[1].vy = random.choice([-2, 2])  # Keep vertical motion

    collision_mode = config["collision_mode"]
    anomaly_frame = SEQ_LEN // 2
    frames = []

    for t in range(SEQ_LEN):
        # ---- Anomaly injection ----
        if config.get("anomaly") and t == anomaly_frame:
            if config["anomaly"] == "slide":
                collision_mode = "slide"
            elif config["anomaly"] == "disappear":
                # Make first active object disappear (2 → 1)
                for obj in objects:
                    if obj.active:
                        obj.active = False
                        break
            elif config["anomaly"] == "appear":
                for obj in objects:
                    if not obj.active:
                        obj.active = True

                        # Try multiple times to find non-overlapping position
                        for _ in range(10):
                            obj.x = random.randint(0, IMG_SIZE - DIGIT_SIZE)
                            obj.y = random.randint(0, IMG_SIZE - DIGIT_SIZE)

                            if not any(
                                overlap(obj, other)
                                for other in objects
                                if other.active and other is not obj
                            ):
                                break

                        obj.vx = random.choice([-2, 2])
                        obj.vy = random.choice([-2, 2])

                        break

        # ---- Step simulation ----
        for obj in objects:
            obj.step(config["motion"], collision_mode)

        # ---- Handle collisions between all active objects ----
        handle_object_collision(objects, collision_mode)
            
        frames.append(render(objects))

    return np.stack(frames)

