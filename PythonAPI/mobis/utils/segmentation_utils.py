import numpy as np
import cv2
import sys
import glob
import os

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc


def fill_holes(image):
    open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('image', open_cv_image)
    # cv2.waitKey(0)
    im_floodfill = open_cv_image.copy()
    cv2.floodFill(im_floodfill, None, (0, 0), 255)
    im_floodfill = cv2.bitwise_not(im_floodfill)
    cv2.bitwise_or(im_floodfill, open_cv_image, im_floodfill)
    return im_floodfill


def is_visible(dynamic_object, resolution, depth):
    bounding_box = dynamic_object['bounding_box']
    if not dynamic_object['visible']:
        return False

    image_box = dynamic_object['camera_box']
    points = [(int(image_box[i, 0]), int(image_box[i, 1])) for i in range(8)]
    height_fail = [all(0 > points[i][1] or points[i][1] > resolution[0] for i in range(8))]
    width_fail = [all(0 > points[i][0] or points[i][0] > resolution[1] for i in range(8))]

    if width_fail[0] or height_fail[0]:
        # print('continue fail')
        return False

    height_fine = [all(0 < points[i][1] < resolution[0] for i in range(8))]
    width_fine = [all(0 < points[i][0] < resolution[1] for i in range(8))]

    if height_fine[0] and width_fine[0]:
        visible = [all(1000 * depth[points[i][1], points[i][0]] < np.linalg.norm(
            np.array([bounding_box[0, i], bounding_box[1, i], bounding_box[2, i]])) for i in range(8))]
        if visible[0]:
            return False

        return True


def is_point_inside_3d_box(bounding_box, point):
    p0 = np.array([bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]])
    p1 = np.array([bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 1]])
    p3 = np.array([bounding_box[0, 3], bounding_box[1, 3], bounding_box[2, 3]])
    p4 = np.array([bounding_box[0, 4], bounding_box[1, 4], bounding_box[2, 4]])

    u = p0 - p1
    v = p0 - p3
    w = p0 - p4
    # first = np.dot(u, np.transpose(point)) <= max(np.dot(u, np.transpose(p0)), np.dot(u, np.transpose(p1))) and np.dot(u, np.transpose(point)) >= min(np.dot(u, np.transpose(p0)), np.dot(u, np.transpose(p1)))
    first = ( np.dot(u, np.transpose(p0)) <= np.dot(u, np.transpose(point)) <= np.dot(u, np.transpose(p1)) ) or ( np.dot(u, np.transpose(p0)) >= np.dot(u, np.transpose(point)) >= np.dot(u, np.transpose(p1)) )
    second = ( np.dot(v, np.transpose(p0)) <= np.dot(v, np.transpose(point)) <= np.dot(v, np.transpose(p3)) ) or ( np.dot(v, np.transpose(p0)) >= np.dot(v, np.transpose(point)) >= np.dot(v, np.transpose(p3)) )
    third = ( np.dot(w, np.transpose(p0)) <= np.dot(w, np.transpose(point)) <= np.dot(w, np.transpose(p4)) ) or ( np.dot(w, np.transpose(p0)) >= np.dot(w, np.transpose(point)) >= np.dot(w, np.transpose(p4)) )
    return_value = (first and second and third)
    return return_value


def convert_carla_img_to_arrays(img, type):
    output_array = None
    if type == 'sseg':
        img.convert(cc.CityScapesPalette)
        output_array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
        output_array = np.reshape(output_array, (img.height, img.width, 4))
        output_array = output_array[:, :, :3]
        output_array = output_array[:, :, ::-1]
    elif type == 'depth':
        output_array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
        output_array = np.reshape(output_array, (img.height, img.width, 4))
        output_array = output_array.astype(np.float32)

    elif type == 'color':
        pass
    else:
        raise('current type %s is not provided', type)

    return output_array


def produce_instance_seg(sseg_img, depth_img, camera, dynamic_objects, resolution):
    array_sseg = convert_carla_img_to_arrays(sseg_img, 'sseg')

    car_indices = np.where(np.all(array_sseg == np.array([0,0,142]), axis=-1))
    pedo_indices = np.where(np.all(array_sseg == np.array([220, 20, 60]), axis=-1))
    indices = [np.concatenate([car_indices[0], pedo_indices[0]]), np.concatenate([car_indices[1], pedo_indices[1]])]

    array_depth = convert_carla_img_to_arrays(depth_img, 'depth')
    normalized_depth = np.dot(array_depth[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    far = 1000.0  # max depth in meters.

    points2d_with_dynamic_objects = np.array([indices[1], indices[0], np.ones(shape=(indices[0].shape))])
    points3d_with_dynamic_objects = np.dot(np.linalg.inv(camera.calibration), points2d_with_dynamic_objects)
    points3d_with_dynamic_objects *= normalized_depth[indices] * far

    instance_seg_output = array_sseg.copy()

    visible_bb = {}
    instance_speed_dict = {}

    for key in dynamic_objects.keys():
        if is_visible(dynamic_objects[key], resolution, normalized_depth):
            visible_bb[key] = dynamic_objects[key]

    for i in range(points3d_with_dynamic_objects.shape[1]):
        for ind, vis_keys in enumerate(visible_bb.keys()):
            inst_id = ind + 1
            color = np.array([inst_id, inst_id, inst_id])
            bounding_box = visible_bb[vis_keys]['bounding_box']
            speed = visible_bb[vis_keys]['speed']

            if is_point_inside_3d_box(bounding_box, np.array([points3d_with_dynamic_objects[0, i],
                                                              points3d_with_dynamic_objects[1, i],
                                                              points3d_with_dynamic_objects[2, i]])):

                instance_seg_output[int(points2d_with_dynamic_objects[1][i]),
                                    int(points2d_with_dynamic_objects[0][i])] = color

                instance_speed_dict[inst_id] = speed

    return instance_seg_output, instance_speed_dict


def produce_mod_segmentation(sseg_img, depth_img, camera, dynamic_objects, resolution):
    instance_seg_img, instance_speed_dict = produce_instance_seg(sseg_img, depth_img, camera, dynamic_objects, resolution)
    mod_segmentation = instance_seg_img.copy()
    mod_segmentation[:, :, :] = np.array([0, 0, 0])

    for instance in instance_speed_dict.keys():
        if instance_speed_dict[instance] > 0.5:
            instance_ind = np.where(np.all(instance_seg_img == np.array([instance, instance, instance]), axis=-1))
            mod_segmentation[instance_ind] = np.array([255, 255, 255])

    return fill_holes(mod_segmentation), instance_seg_img
