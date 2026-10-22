"""Privileged geometry observations, explicitly separate from pixel benchmarks."""
import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

ROAD_STATE_NAMES = (
    'forward_speed', 'lateral_speed', 'angular_speed', 'cross_track_offset',
    'tangent_right', 'tangent_forward', 'waypoint3_right', 'waypoint3_forward',
    'waypoint8_right', 'waypoint8_forward', 'waypoint16_right', 'waypoint16_forward',
    'coverage', 'elapsed_fraction', 'front_wheel_angle', 'onroad_wheel_fraction',
)
ROAD_STATE_DIM = len(ROAD_STATE_NAMES)


def road_features(base, centers, raw_steps):
    car = base.car
    position = np.asarray(car.hull.position)
    velocity = np.asarray(car.hull.linearVelocity)
    angle = car.hull.angle
    right = np.array([np.cos(angle), np.sin(angle)])
    forward = np.array([-np.sin(angle), np.cos(angle)])
    index = int(np.square(centers-position).sum(1).argmin())
    tangent = centers[(index+1) % len(centers)]-centers[(index-1) % len(centers)]
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    normal = np.array([tangent[1], -tangent[0]])
    values = [velocity@forward/60, velocity@right/30, car.hull.angularVelocity/4,
              (position-centers[index])@normal/TRACK_WIDTH, tangent@right, tangent@forward]
    for offset, scale in ((3, 20), (8, 40), (16, 70)):
        delta = centers[(index+offset) % len(centers)]-position
        values.extend([delta@right/scale, delta@forward/scale])
    values.extend([base.tile_visited_count/len(centers), raw_steps/1000,
                   car.wheels[0].joint.angle/.5,
                   sum(bool(w.tiles) for w in car.wheels)/4])
    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)


def state_plane(features, size):
    plane = np.full((size, size), 128, dtype=np.uint8)
    plane.flat[:ROAD_STATE_DIM] = np.rint(features*127+128).astype(np.uint8)
    return plane
