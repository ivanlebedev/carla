import numpy as np
import cv2
import carla
import pygame
# from constants import *



def compute_2d_bounding_boxes(instance_sseg, dynamic_objects):
    instance_2d_bb_dict = {}
    for ind, key in enumerate(dynamic_objects.keys()):
        color = np.array([ind, ind, ind])
        active_px = np.where(np.all(instance_sseg == color, axis=-1))
        active_px = list(zip(np.array(active_px[1]), np.array(active_px[0])))
        if active_px:
            x, y, w, h = cv2.boundingRect(np.array(active_px))
            instance_2d_bb_dict[key] = [x, y, w, h]
    return instance_2d_bb_dict


def draw_2d_bounding_boxes(color_image, instance_sseg, dynamic_objects):
    rgb_array = np.frombuffer(color_image.raw_data, dtype=np.dtype("uint8"))
    rgb_array = np.reshape(rgb_array, (color_image.height, color_image.width, 4))
    rgb_array = rgb_array[:, :, :3]
    rgb_array = rgb_array[:, :, ::-1]

    img_with_2d_bb = rgb_array.copy()
    for ind, key in enumerate(dynamic_objects.keys()):
        color = np.array([ind, ind, ind])
        active_px = np.where(np.all(instance_sseg == color, axis=-1))
        active_px = list(zip(np.array(active_px[1]), np.array(active_px[0])))
        if active_px:
            x, y, w, h = cv2.boundingRect(np.array(active_px))
            cv2.rectangle(img_with_2d_bb, (x, y), (x + w, y + h), (3, 255, 0), 1)
    return img_with_2d_bb



def get_3d_boxes(vehicles, camera):
    """
    Creates 3D bounding boxes in world coordinate system
    :param vehicles:
    :param camera:
    :return:
    """
    bounding_boxes = [get_3d_box(vehicle, camera) for vehicle in vehicles]
    # # filter objects behind camera
    # bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
    return bounding_boxes


def get_bounding_boxes(vehicles, camera):
    """
    Creates 3D bounding boxes based on carla vehicle list and camera.
    """

    bounding_boxes = [get_bounding_box(vehicle, camera) for vehicle in vehicles]
    # filter objects behind camera
    bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
    return bounding_boxes


def draw_bounding_boxes(display, bounding_boxes):
    """
    Draws bounding boxes on pygame display.
    """

    bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
    bb_surface.set_colorkey((0, 0, 0))
    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        # draw lines
        # base
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
        # top
        pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
        pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
        # base-top
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
    display.blit(bb_surface, (0, 0))


def world_bb_to_camera_bb(bb, camera):
    bbox = np.transpose(np.dot(camera.calibration, bb))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    return camera_bbox


def get_3d_box(vehicle, camera):
    bb_cords = _create_bb_points(vehicle)
    cords_x_y_z = _vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    return cords_y_minus_z_x


def get_bounding_box(vehicle, camera):
    """
    Returns 3D bounding box for a vehicle based on camera view.
    """

    bb_cords = _create_bb_points(vehicle)
    cords_x_y_z = _vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    return camera_bbox


def _create_bb_points(vehicle):
    """
    Returns 3D bounding box for a vehicle.
    """

    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    extent = 1.1 * extent
    extent.x = 1.5 * extent.x
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords


def _vehicle_to_sensor(cords, vehicle, sensor):
    """
    Transforms coordinates of a vehicle bounding box to sensor.
    """

    world_cord = _vehicle_to_world(cords, vehicle)
    sensor_cord = _world_to_sensor(world_cord, sensor)
    return sensor_cord


def _vehicle_to_world(cords, vehicle):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """

    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords


def _world_to_sensor(cords, sensor):
    """
    Transforms world coordinates to sensor.
    """

    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords


def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

