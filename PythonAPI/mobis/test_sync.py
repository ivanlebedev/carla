#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import logging
import random
import datetime

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

def saveimage_id(path, image, converter, id):
    filepath = os.path.join(path, '%08d.png' % image.frame_number)
    # saveimageobj(image, filepath)
    image.save_to_disk(filepath, converter)

def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def getcurtimestr():
    currentDT = datetime.datetime.now()
    return (currentDT.strftime("%Y%m%d_%H%M%S"))

def setup_camera(camera):
    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    camera.calibration = calibration


def main():
    foldername = os.path.join('./_out', getcurtimestr())
    actor_list = []
    pygame.init()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    print('enabling synchronous mode.')
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    try:
        m = world.get_map()
        print('beginning of try')
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera)

        camerad = world.spawn_actor(
            blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camerad)

        camerasseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camerasseg)

        # Make sync queue for sensor data.
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # Make sync queue for sensor data.
        image_queued = queue.Queue()
        camerad.listen(image_queued.put)

        # Make sync queue for sensor data.
        image_queuesseg = queue.Queue()
        camerasseg.listen(image_queuesseg.put)

        frame = None

        # display = pygame.display.set_mode(
        #     (800, 600),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        # font = get_font()

        clock = pygame.time.Clock()

        while True:
            if should_quit():
                return

            clock.tick()
            world.tick()
            ts = world.wait_for_tick()

            if frame is not None:
                if ts.frame_count != frame + 1:
                    logging.warning('frame skip!')

            frame = ts.frame_count

            while True:
                image = image_queue.get()
                imaged = image_queued.get()
                imagesseg = image_queuesseg.get()

                if image.frame_number == ts.frame_count and imaged.frame_number == ts.frame_count and imagesseg.frame_number == ts.frame_count:
                    break
                logging.warning(
                    'wrong image time-stampstamp: frame=%d, image.frame=%d, imaged.frame=%d, imagesseg.frame=%d',
                    ts.frame_count,
                    image.frame_number, imaged.frame_number, imagesseg.frame_number)

            waypoint = random.choice(waypoint.next(2))
            vehicle.set_transform(waypoint.transform)

            path1 = os.path.join(foldername, 'rgb')
            if not os.path.exists(path1): os.makedirs(path1)
            cc_raw = carla.ColorConverter.Raw
            saveimage_id(path1, image, cc_raw, camera.id)

            path2 = os.path.join(foldername, 'depth')
            if not os.path.exists(path2): os.makedirs(path2)
            cc_depth = carla.ColorConverter.LogarithmicDepth
            saveimage_id(path2, imaged, cc_depth, camerad.id)

            path3 = os.path.join(foldername, 'seg')
            if not os.path.exists(path3): os.makedirs(path3)
            cc_seg = carla.ColorConverter.CityScapesPalette
            saveimage_id(path3, imagesseg, cc_seg, camerasseg.id)

            #draw_image(display, image)

            #text_surface = font.render('% 5d FPS' % clock.get_fps(), True, (255, 255, 255))
            #display.blit(text_surface, (8, 10))

            #pygame.display.flip()

    finally:
        print('\ndisabling synchronous mode.')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    main()
