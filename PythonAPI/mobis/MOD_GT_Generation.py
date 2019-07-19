# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""
import weakref
import random
import numpy as np
import datetime
import copy
#
import sys
sys.path.append("utils")
from segmentation_utils import *
from bounding_box_utils import *
# from carla.Util.bounding_box_utils import *


import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w

import time

import glob
import os
import sys
import math
from PIL import Image
import cv2


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


try:
    import queue
except ImportError:
    import Queue as queue


import carla
from carla import ColorConverter as cc

#
# VIEW_WIDTH = 1226
# VIEW_HEIGHT = 370
# VIEW_FOV = 81.846128

# mtck data
VIEW_WIDTH = 2048
VIEW_HEIGHT = 1024
VIEW_FOV = 59.54

# # mtck data
# VIEW_WIDTH = 1024
# VIEW_HEIGHT = 512
# VIEW_FOV = 59.54

# LIDAR
CHANNELS = 64
RANGE = 1000
POINTS_PER_SECOND = 56000
ROTATION_FREQUENCY = 10


#
# class ClientSideInstanceSeg(object):
#
#     @staticmethod
#     def compute_2d_bounding_boxes(color_image, instance_sseg, bb_dict):
#         arrayi = np.frombuffer(color_image.raw_data, dtype=np.dtype("uint8"))
#         arrayi = np.reshape(arrayi, (color_image.height, color_image.width, 4))
#         arrayi = arrayi[:, :, :3]
#         arrayi = arrayi[:, :, ::-1]
#         img69 = arrayi.copy()
#         for k in bb_dict.keys():
#             color = np.array([k, 200, 250 - k])
#             active_px = np.where(np.all(instance_sseg == color, axis=-1))
#             active_px = list(zip(np.array(active_px[1]), np.array(active_px[0])))
#             if active_px:
#                 x, y, w, h = cv2.boundingRect(np.array(active_px))
#                 cv2.rectangle(img69, (x, y), (x + w, y + h), (3, 255, 0), 1)
#         return img69
#
#     @staticmethod
#     def fill_holes(image):
#         open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         im_floodfill = open_cv_image.copy()
#         cv2.floodFill(im_floodfill, None, (0, 0), 255)
#         im_floodfill = cv2.bitwise_not(im_floodfill)
#         cv2.bitwise_or(im_floodfill, open_cv_image, im_floodfill)
#         return im_floodfill
#
#     @staticmethod
#     def is_visible(bounding_box, depth_img):
#         pass
#
#     @staticmethod
#     def is_point_inside_3d_box(bounding_box, point):
#         p0 = np.array([bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]])
#         p1 = np.array([bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 1]])
#         p3 = np.array([bounding_box[0, 3], bounding_box[1, 3], bounding_box[2, 3]])
#         p4 = np.array([bounding_box[0, 4], bounding_box[1, 4], bounding_box[2, 4]])
#
#         u = p0 - p1
#         v = p0 - p3
#         w = p0 - p4
#
#         first = (np.dot(u, np.transpose(p0)) <= np.dot(u, np.transpose(point)) <= np.dot(u, np.transpose(p1))) or \
#                 (np.dot(u, np.transpose(p0)) >= np.dot(u, np.transpose(point)) >= np.dot(u, np.transpose(p1)))
#
#         second = (np.dot(v, np.transpose(p0)) <= np.dot(v, np.transpose(point)) <= np.dot(v, np.transpose(p3))) or \
#                  (np.dot(v, np.transpose(p0)) >= np.dot(v, np.transpose(point)) >= np.dot(v, np.transpose(p3)))
#
#         third = (np.dot(w, np.transpose(p0)) <= np.dot(w, np.transpose(point)) <= np.dot(w, np.transpose(p4))) or \
#                 (np.dot(w, np.transpose(p0)) >= np.dot(w, np.transpose(point)) >= np.dot(w, np.transpose(p4)))
#
#         return_value = (first and second and third)
#         return return_value
#
#     @staticmethod
#     def produce_instance_seg(sseg_img, depth_img, camera, img, dynamic_objects, save_mod, bb_dict):
#         start = time.time()
#
#         sseg_img.convert(cc.CityScapesPalette)
#         array_sseg = np.frombuffer(sseg_img.raw_data, dtype=np.dtype("uint8"))
#         array_sseg = np.reshape(array_sseg, (sseg_img.height, sseg_img.width, 4))
#         array_sseg = array_sseg[:, :, :3]
#         array_sseg = array_sseg[:, :, ::-1]
#
#         car_indices = np.where(np.all(array_sseg == np.array([0,0,142]), axis=-1))
#         pedo_indices = np.where(np.all(array_sseg == np.array([220, 20, 60]), axis=-1))
#
#         indices = [np.concatenate([car_indices[0], pedo_indices[0]]), np.concatenate([car_indices[1], pedo_indices[1]])]
#
#         arrayd = np.frombuffer(depth_img.raw_data, dtype=np.dtype("uint8"))
#         arrayd = np.reshape(arrayd, (depth_img.height, depth_img.width, 4))
#         arrayd = arrayd.astype(np.float32)
#         normalized_depth = np.dot(arrayd[:, :, :3], [65536.0, 256.0, 1.0])
#         normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
#         far = 1000.0  # max depth in meters.
#
#         p2d = np.array([indices[1], indices[0], np.ones(shape=indices[0].shape)])
#         p3d = np.dot(np.linalg.inv(camera.calibration), p2d)
#         p3d *= normalized_depth[indices] * far
#
#         img2 = array_sseg.copy()
#         img3 = array_sseg.copy()
#         img3[:,:,:] = np.array([0,0,0])
#
#         # after_depth = time.time()
#         # print('after depth', after_depth - start)
#
#         visible_bb = {}
#
#         for key in dynamic_objects.keys():
#             # for statistics
#             moving = False
#             any = False
#
#             # start2 = time.time()
#             bounding_box = dynamic_objects[key]['bounding_box']
#             if not dynamic_objects[key]['visible']:
#                 # print('continue not visible')
#                 continue
#
#             image_box = dynamic_objects[key]['camera_box']
#             points = [(int(image_box[i, 0]), int(image_box[i, 1])) for i in range(8)]
#             # if all([normalized_depth[points[i]] < np.linalg.norm(bounding_box[i]) for i in range(8)]):
#             height_fail = [all(0 > points[i][1] or points[i][1] > VIEW_HEIGHT for i in range(8))]
#             width_fail = [all(0 > points[i][0] or points[i][0] > VIEW_WIDTH for i in range(8))]
#
#             if width_fail[0] or height_fail[0]:
#                 # print('continue fail')
#                 continue
#
#             height_fine = [all(0 < points[i][1] < VIEW_HEIGHT for i in range(8))]
#             width_fine = [all(0 < points[i][0] < VIEW_WIDTH for i in range(8))]
#
#             if height_fine[0] and width_fine[0]:
#                 visible = [all(1000 * normalized_depth[points[i][1], points[i][0]] < np.linalg.norm(
#                         np.array([bounding_box[0, i], bounding_box[1, i], bounding_box[2, i]])) for i in range(8))]
#                 if visible[0]:
#                     # print('continue second fail')
#                     continue
#
#             visible_bb[key] = dynamic_objects[key]
#
#         bb_dict['num_cars_bb'] += dynamic_objects.__len__()
#         bb_dict['num_moving_bb'] += visible_bb.__len__()
#
#         for i in range(p3d.shape[1]):
#             for vis_keys in visible_bb.keys():
#
#                 bounding_box = visible_bb[vis_keys]['bounding_box']
#                 speed = visible_bb[vis_keys]['speed']
#
#                 if ClientSideInstanceSeg.is_point_inside_3d_box(bounding_box, np.array([p3d[0, i], p3d[1, i], p3d[2, i]])):
#                     img2[int(p2d[1][i]), int(p2d[0][i])] = np.array([key, 200, 250 - key])
#
#                     if save_mod and speed > 0.5:
#                         img3[int(p2d[1][i]), int(p2d[0][i])] = np.array([255, 255, 255])
#
#         end = time.time()
#         print('total time', end - start)
#         return img2, ClientSideInstanceSeg.fill_holes(img3)
#
# # ==============================================================================
# # -- ClientSideBoundingBoxes ---------------------------------------------------
# # ==============================================================================
#
#
# class ClientSideBoundingBoxes(object):
#     """
#     This is a module responsible for creating 3D bounding boxes and drawing them
#     client-side on pygame surface.
#     """
#     @staticmethod
#     def get_3d_boxes(vehicles, camera):
#         """
#         Creates 3D bounding boxes in world coordinate system
#         :param vehicles:
#         :param camera:
#         :return:
#         """
#         bounding_boxes = [ClientSideBoundingBoxes.get_3d_box(vehicle, camera) for vehicle in vehicles]
#         # # filter objects behind camera
#         # bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
#         return bounding_boxes
#
#     @staticmethod
#     def get_bounding_boxes(vehicles, camera):
#         """
#         Creates 3D bounding boxes based on carla vehicle list and camera.
#         """
#
#         bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
#         # filter objects behind camera
#         bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
#         return bounding_boxes
#
#     @staticmethod
#     def draw_bounding_boxes(display, bounding_boxes):
#         """
#         Draws bounding boxes on pygame display.
#         """
#
#         bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
#         bb_surface.set_colorkey((0, 0, 0))
#         for bbox in bounding_boxes:
#             points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
#             # draw lines
#             # base
#             pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
#             pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
#             pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
#             pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
#             # top
#             pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
#             pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
#             pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
#             pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
#             # base-top
#             pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
#             pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
#             pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
#             pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
#         display.blit(bb_surface, (0, 0))
#
#
#     @staticmethod
#     def world_bb_to_camera_bb(bb, camera):
#         bbox = np.transpose(np.dot(camera.calibration, bb))
#         camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
#         return camera_bbox
#
#     @staticmethod
#     def get_3d_box(vehicle, camera):
#         bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
#         cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
#         cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
#         return cords_y_minus_z_x
#
#     @staticmethod
#     def get_bounding_box(vehicle, camera):
#         """
#         Returns 3D bounding box for a vehicle based on camera view.
#         """
#
#         bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
#         cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
#         cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
#         bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
#         camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
#         return camera_bbox
#
#     @staticmethod
#     def _create_bb_points(vehicle):
#         """
#         Returns 3D bounding box for a vehicle.
#         """
#
#         cords = np.zeros((8, 4))
#         extent = vehicle.bounding_box.extent
#         extent = 1.1 * extent
#         extent.x = 1.5 * extent.x
#         cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
#         cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
#         cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
#         cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
#         cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
#         cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
#         cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
#         cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
#         return cords
#
#     @staticmethod
#     def _vehicle_to_sensor(cords, vehicle, sensor):
#         """
#         Transforms coordinates of a vehicle bounding box to sensor.
#         """
#
#         world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
#         sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
#         return sensor_cord
#
#     @staticmethod
#     def _vehicle_to_world(cords, vehicle):
#         """
#         Transforms coordinates of a vehicle bounding box to world.
#         """
#
#         bb_transform = carla.Transform(vehicle.bounding_box.location)
#         bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
#         vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
#         bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
#         world_cords = np.dot(bb_world_matrix, np.transpose(cords))
#         return world_cords
#
#     @staticmethod
#     def _world_to_sensor(cords, sensor):
#         """
#         Transforms world coordinates to sensor.
#         """
#
#         sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
#         world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
#         sensor_cords = np.dot(world_sensor_matrix, cords)
#         return sensor_cords
#
#     @staticmethod
#     def get_matrix(transform):
#         """
#         Creates matrix from carla transform.
#         """
#
#         rotation = transform.rotation
#         location = transform.location
#         c_y = np.cos(np.radians(rotation.yaw))
#         s_y = np.sin(np.radians(rotation.yaw))
#         c_r = np.cos(np.radians(rotation.roll))
#         s_r = np.sin(np.radians(rotation.roll))
#         c_p = np.cos(np.radians(rotation.pitch))
#         s_p = np.sin(np.radians(rotation.pitch))
#         matrix = np.matrix(np.identity(4))
#         matrix[0, 3] = location.x
#         matrix[1, 3] = location.y
#         matrix[2, 3] = location.z
#         matrix[0, 0] = c_p * c_y
#         matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
#         matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
#         matrix[1, 0] = s_y * c_p
#         matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
#         matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
#         matrix[2, 0] = s_p
#         matrix[2, 1] = -c_p * s_r
#         matrix[2, 2] = c_p * c_r
#         return matrix
#

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.map = None
        self.camera = None
        self.camera_depth = None
        self.camera_sseg = None
        self.car = None

        self.actor_list = []
        self.pedo_list = []

        self.display = None

        self.image = None
        self.image_depth = None
        self.image_sseg = None

        self.image_queue = queue.Queue()
        self.image_depth_queue = queue.Queue()
        self.image_sseg_queue = queue.Queue()

        self.bb_dict = dict()
        self.draw_bb = False
        self.store_stuff = True
        self.save_mod = self.store_stuff and True
        self.save_instance = self.store_stuff and True
        self.save_all = self.store_stuff and False
        self.draw_2d_bb = False

        self.autopilot = False

        # statistics
        self.bb_dict = dict()
        self.bb_dict['num_cars_bb'] = 0
        self.bb_dict['num_moving_bb'] = 0
        self.frame_num = 0

    @staticmethod
    def getcurtimestr():
        currentDT = datetime.datetime.now()
        return (currentDT.strftime("%Y%m%d_%H%M%S"))

    @staticmethod
    def saveimage_id(path, image, converter, id):
        filepath = os.path.join(path, '%06d.png' % id)
        # saveimageobj(image, filepath)
        image.save_to_disk(filepath, converter)

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def camera_depth_blueprint(self):
        """
        Returns depth blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def camera_sseg_blueprint(self):
        """
        Returns sseg blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def camera_lidar_blueprint(self):
        """
        Returns lidar blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        camera_bp.set_attribute('channels', str(CHANNELS))
        camera_bp.set_attribute('range', str(RANGE))
        camera_bp.set_attribute('points_per_second', str(POINTS_PER_SECOND))
        camera_bp.set_attribute('rotation_frequency', str(ROTATION_FREQUENCY))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)



    def setup_car(self, transform):
        """
        Spawns actor-vehicle to be controled.
        """

        # transform = carla.Transform(carla.Location(x=8, y=0, z=1), carla.Rotation(yaw=90))
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[5]
        self.car = self.world.spawn_actor(car_bp, transform)
        # if self.autopilot:
        #     vehicle_control = self.car.get_control()
        #     vehicle_control.throttle = 1.0 if self.autopilot else 0.0
        #     print('movingtrue')
        #     self.car.apply_control(vehicle_control)

        self.car.set_autopilot(self.autopilot)
        self.actor_list.append(self.car)

    def setup_camera(self, attach_to=None):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        # camera_transform = carla.Transform(carla.Location(x=-80.8, y=170.5, z=2.8), carla.Rotation(pitch=-15, yaw=-90))
        camera_transform = carla.Transform(carla.Location(x=0.2, z=1.52), carla.Rotation(pitch=1))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=attach_to)
        self.camera.listen(self.image_queue.put)

        if self.store_stuff:
            self.camera_depth = self.world.spawn_actor(self.camera_depth_blueprint(), camera_transform, attach_to=attach_to)
            self.camera_sseg = self.world.spawn_actor(self.camera_sseg_blueprint(), camera_transform, attach_to=attach_to)

            self.camera_depth.listen(self.image_depth_queue.put)
            self.camera_sseg.listen(self.image_sseg_queue.put)

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        if self.autopilot:
            return False

        control = car.get_control()
        control.throttle = 0.0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def create_pedestrian(self, location):
        blueprints = self.world.get_blueprint_library().filter('walker.*')

        def try_spawn_random_pedo_at(transform):
            blueprint = random.choice(blueprints)
            pedestrian_heading = random.randint(0, 360)
            player = self.world.try_spawn_actor(blueprint, transform)
            if player is not None:
                self.actor_list.append(player)
                player_control = carla.WalkerControl()
                player_control.speed = random.randint(0, 4)
                # player_control.speed = 0
                player_rotation = carla.Rotation(0, pedestrian_heading, 0)
                player_control.direction = player_rotation.get_forward_vector()
                player.apply_control(player_control)
                self.pedo_list.append(player)
                # print('spawned %r at %s' % (player.type_id, transform.location))
                return True
            return False

        try_spawn_random_pedo_at(location)

    def add_vehicle(self, ind, moving=False, transform=None, bp=None, direction=None, speed=1.0):
        if not transform:
            spawn_points = self.map.get_spawn_points()
            transform = random.choice(spawn_points) if spawn_points else carla.Transform()
        if not bp:
            bp = self.world.get_blueprint_library().filter('vehicle.*')[ind]

        color = (bp.get_attribute('color').recommended_values)[0]
        bp.set_attribute('color', color)
        vehicle = self.world.try_spawn_actor(bp, transform)

        if vehicle:
            vehicle_control = carla.VehicleControl()
            vehicle_control.throttle = speed if moving else 0.0

            vehicle.apply_control(vehicle_control)
        self.actor_list.append(vehicle)

    def spawn_npc(self, num):
        spawn_points = self.world.get_map().get_spawn_points()
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('model3')]

        random.shuffle(spawn_points)

        for n, transform in enumerate(spawn_points):
            if n >= num:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle:
                vehicle.set_autopilot(True)

                self.actor_list.append(vehicle)

    def spawn_bunch_of_pedos(self, number, transforma):
        print('initial location x: {}, y: {}, z: {}'.format(transforma.location.x, transforma.location.y, transforma.location.z))

        al = np.radians(transforma.rotation.yaw)
        trans = np.array([transforma.location.x, transforma.location.y])
        rot_mat = np.zeros((2, 2))
        rot_mat[0, 0] = math.cos(al)
        rot_mat[0, 1] = -math.sin(al)
        rot_mat[1, 0] = math.sin(al)
        rot_mat[1, 1] = math.cos(al)

        for i in range(number):
            # transformnew = transforma

            # value_x = random.randint(2, 25)
            # value_y = random.randint(-25, 25)
            pos = np.array([random.randint(5, 60), random.randint(-15, 15)])
            # pos = np.array([10, 2])
            pos = rot_mat.dot(pos)
            pos = pos + trans
            transformnew = carla.Transform(
                carla.Location(x=pos[0], y=pos[1], z=transforma.location.z + 1.5))
            # print('x: {}, y: {}, x_des: {}, y_des: {}'.format(value_x, value_y, decision_x, decision_y))
            # transformnew.location.x += value_x if decision_x else -value_x
            # transformnew.location.y += value_y if decision_y else -value_y
            print('new location x: {}, y: {}, z: {}'.format(transformnew.location.x, transformnew.location.y,
                                                                transformnew.location.z))
            self.create_pedestrian(transformnew)


    def spawn_static_guys_for_matteo(self, location_car):
        xcar = location_car.location.x
        ycar = location_car.location.y
        transform1 = carla.Transform(
            carla.Location(x=xcar - 1.5, y=ycar - 5, z=2))
        self.create_pedestrian(transform1)

        transform2 = carla.Transform(
            carla.Location(x=xcar - 1.5, y=ycar - 10, z=2))
        self.create_pedestrian(transform2)

        transform3 = carla.Transform(
            carla.Location(x=xcar - 1.5, y=ycar - 15, z=2))
        self.create_pedestrian(transform3)

        transform4 = carla.Transform(
            carla.Location(x=xcar + 1.5, y=ycar - 5, z=2))
        self.create_pedestrian(transform4)

        transform5 = carla.Transform(
            carla.Location(x=xcar + 1.5, y=ycar - 10, z=2))
        self.create_pedestrian(transform5)

        transform6 = carla.Transform(
            carla.Location(x=xcar + 1.5, y=ycar - 15, z=2))
        self.create_pedestrian(transform6)

        transform7 = carla.Transform(
            carla.Location(x=xcar + 1.5, y=ycar - 25, z=2))
        self.create_pedestrian(transform7)

        transform8 = carla.Transform(
            carla.Location(x=xcar - 1.5, y=ycar - 25, z=2))
        self.create_pedestrian(transform8)


        transform9 = carla.Transform(
            carla.Location(x=xcar + 1.5, y=ycar - 50, z=2))
        self.create_pedestrian(transform9)

        transform10 = carla.Transform(
            carla.Location(x=xcar - 1.5, y=ycar - 50, z=2))
        self.create_pedestrian(transform10)



    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            num_of_frames_for_reset = -1

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            # self.world = self.client.load_world('Town04')
            self.map = self.client.get_world().get_map()

            self.store_stuff = True
            self.save_mod = self.store_stuff and True

            spawn_points = self.world.get_map().get_spawn_points()

            save_folder = os.path.join('/home/mobis/bigdi/CarlaK', BasicSynchronousClient.getcurtimestr())

            transformnew = carla.Transform(
                carla.Location(x=5.245205402374, y=129.31069946289062, z=1), carla.Rotation(yaw=-90))

            transformnew2 = carla.Transform(
                carla.Location(x=5.245205402374, y=110.31069946289062, z=1), carla.Rotation(yaw=-90))

            self.add_vehicle(5, moving=False, transform=transformnew2, bp=None, direction=None, speed=0.0)

            # self.setup_car(spawn_points[1])
            self.setup_car(transformnew)
            self.setup_camera(attach_to=self.car)



            # self.spawn_static_guys_for_matteo(transformnew)

            # pedos
            num_pedos = random.randint(7,20)
            # self.spawn_bunch_of_pedos(num_pedos, spawn_points[5])

            # cars
            # self.spawn_npc(50)

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            pedos = self.world.get_actors().filter('walker.*')
            vehicles =  self.world.get_actors().filter('vehicle.*')

            while True:
                dynamic_object_dict_in_this_frame = dict()

                pygame_clock.tick()
                self.world.tick()
                ts = self.world.wait_for_tick()


                if self.store_stuff:
                    """
                    if image saving is active
                    """
                    self.image = self.image_queue.get()
                    self.image_depth = self.image_depth_queue.get()
                    self.image_sseg = self.image_sseg_queue.get()

                    ####################################### SENSOR SYNCHRONIZATION #########################################################

                    while True:
                        if self.image.frame_number == ts.frame_count and self.image_depth.frame_number == ts.frame_count\
                                and self.image_sseg.frame_number == ts.frame_count:
                            break

                        if self.image.frame_number != ts.frame_count:
                            self.image = self.image_queue.get()
                        if self.image_depth.frame_number != ts.frame_count:
                            self.image_depth = self.image_depth_queue.get()
                        if self.image_sseg.frame_number != ts.frame_count:
                            self.image_sseg = self.image_sseg_queue.get()

                    # after_sync = time.time()
                    # print('after sync', after_sync - start)

                    if self.save_all:
                        path2 = os.path.join(save_folder, 'depth')
                        if not os.path.exists(path2): os.makedirs(path2)
                        cc_depth = carla.ColorConverter.LogarithmicDepth
                        self.saveimage_id(path2, self.image_depth, cc_depth, 686)

                        path3 = os.path.join(save_folder, 'seg')
                        if not os.path.exists(path3): os.makedirs(path3)
                        cc_seg = carla.ColorConverter.CityScapesPalette
                        self.saveimage_id(path3, self.image_sseg, cc_seg, 6868)

                    ############################## DICT GENERATION WITH VEHICLE INFORMATION ################################################

                    for vehicle in vehicles:
                        bb = get_3d_box(vehicle, self.camera)
                        camera_bb = world_bb_to_camera_bb(bb, self.camera)
                        # Client
                        # if all(bb[:, 2] > 0):
                        v_transform = vehicle.get_transform()
                        velocity = vehicle.get_velocity()
                        is_visible = all(camera_bb[:, 2] > 0)
                        location_gnss = self.map.transform_to_geolocation(v_transform.location)
                        temp = {
                            "id": vehicle.id,
                            "bounding_box": bb,
                            'camera_box': camera_bb,
                            "speed": (3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)),
                            'location' : location_gnss,
                            'visible' : is_visible
                        }
                        dynamic_object_dict_in_this_frame[vehicle.id] = temp

                    for pedo in self.pedo_list:
                        bb = get_3d_box(pedo, self.camera)
                        camera_bb = world_bb_to_camera_bb(bb, self.camera)
                        is_visible = all(camera_bb[:, 2] > 0)
                        v_transform = pedo.get_transform()
                        velocity = pedo.get_velocity()
                        location_gnss = self.map.transform_to_geolocation(v_transform.location)
                        temp = {
                            "id": pedo.id,
                            "bounding_box": bb,
                            'camera_box': camera_bb,
                            "speed": (3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)),
                            'location': location_gnss,
                            'visible' : is_visible
                        }
                        # print('bounding box human:', bb)
                        dynamic_object_dict_in_this_frame[pedo.id] = temp

                    # after_bb = time.time()
                    # print('after computing bounding boxes', after_bb - after_sync)

                    ########################################## CALCULATIONS ###############################################################
                    mod, instance_seg = produce_mod_segmentation(self.image_sseg, self.image_depth, self.camera,
                                                              dynamic_object_dict_in_this_frame, (VIEW_HEIGHT, VIEW_WIDTH))

                    ########################################## SAVING IMAGES ###############################################################
                    # after_sseg = time.time()
                    # print('after instance seg', after_sseg - after_bb)

                    path1 = os.path.join(save_folder, 'RawImages')
                    if not os.path.exists(path1): os.makedirs(path1)
                    cc_raw = carla.ColorConverter.Raw
                    self.saveimage_id(path1, self.image, cc_raw, self.frame_num)

                    if self.draw_2d_bb:
                        image_with_bb = draw_2d_bounding_boxes(self.image, instance_seg, dynamic_object_dict_in_this_frame)
                        im3 = Image.fromarray(image_with_bb)
                        path6 = os.path.join(save_folder, '2dbb')
                        if not os.path.exists(path6): os.makedirs(path6)
                        filepath = os.path.join(path6, '%06d.png' % self.frame_num)
                        im3.save(filepath)

                    if self.save_instance:
                        im = Image.fromarray(instance_seg)
                        path4 = os.path.join(save_folder, 'instance')
                        if not os.path.exists(path4): os.makedirs(path4)
                        filepath = os.path.join(path4, '%06d.png' % self.frame_num)
                        im.save(filepath)

                    im2 = Image.fromarray(mod)
                    path5 = os.path.join(save_folder, 'ModMask')
                    if not os.path.exists(path5): os.makedirs(path5)
                    filepath = os.path.join(path5, '%06d.png' % self.frame_num)
                    im2.save(filepath)

                    # after_saving = time.time()
                    # print('after saving', after_saving - after_sseg)

                    ######################################################################
                    # self.frame_num += 1
                else:
                    while True:
                        self.image = self.image_queue.get()
                        if self.image.frame_number == ts.frame_count:
                            break

                # trans = self.actor_list[0].get_transform()
                # print('location: x = {}, y = {}, z = {}'.format(trans.location.x, trans.location.y, trans.location.z))
                self.render(self.display)

                if self.draw_bb:
                    bounding_boxes = get_bounding_boxes(vehicles, self.camera)
                    draw_bounding_boxes(self.display, bounding_boxes)

                # print(ClientSideBoundingBoxes.get_matrix(self.car.get_transform())[:, 3])
                pygame.display.flip()
                pygame.event.pump()

                if self.control(self.car):
                    return
                print('frame num: {}'.format(self.frame_num))
                self.frame_num += 1
                if self.frame_num == num_of_frames_for_reset:
                    self.frame_num = 0
                    self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.pedo_list])
                    spawn = vehicles[0].get_transform()
                    transformnew = carla.Transform(
                        carla.Location(x=spawn.location.x, y=spawn.location.y, z=spawn.location.z))
                    self.spawn_bunch_of_pedos(num_pedos, transformnew)
                    save_folder = os.path.join('/home/mobis/bigdi/CarlaK/', self.getcurtimestr())
                    # self.world = self.client.get_world()
                    # pedos = self.world.get_actors().filter('walker.*')



        finally:
            # if self.store_stuff:
            #     stat_file_dir = os.path.join(save_folder, 'stats.txt')
            #     f = open(stat_file_dir, "w+")
            #     f.write(
            #         'statistics for this sequence: \n number of bounding boxes: {}\n number of moving object boxes: {}\n number of frames: {} '.format(
            #             self.bb_dict['num_cars_bb'], self.bb_dict['num_moving_bb'], self.frame_num))
            #     f.close()

            self.set_synchronous_mode(False)
            self.camera.destroy()
            if self.store_stuff:
                self.camera_depth.destroy()
                self.camera_sseg.destroy()
            self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.actor_list])
            # self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()