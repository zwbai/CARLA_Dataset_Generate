#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Open3D Lidar visuialization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import math
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import CMM_CARLA_Config as CFG
from math import pi
import logging
from numpy.linalg import pinv, inv

from CMM_CARLA_Config import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from kitti_data_descriptor import KittiDescriptor



VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("../_out_20hz_50x50label_safe_100v_ry", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'ImageSets']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
INDEX_PATH = os.path.join(OUTPUT_FOLDER, 'ImageSets')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LIDAR_PATH_PLY = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.ply')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')


def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:

            lidar_bp.set_attribute('noise_stddev', '0.002')
            lidar_bp.set_attribute('dropoff_general_rate', '0.001')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.05))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1/delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def creat_carla_bbox(point_cloud, point_list, world, frame, lidar):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

    carla_world = world
    vehicle_bbox_list = carla_world.get_level_bbs(actor_type=carla.libcarla.CityObjectLabel.Vehicles)
    pedestrians_bbox_list = carla_world.get_level_bbs(actor_type=carla.libcarla.CityObjectLabel.Pedestrians)
    # print("vehicle_bbox_list", vehicle_bbox_list)
    i = 0
    j = 0
    label_data = []
    for actor in pedestrians_bbox_list:
        # extent.x extent.y extent.z
        # location.x location.y location.z
        # rotation.pitch rotation.yaw rotation.roll
        locationX = round(actor.location.x, 2)
        locationY = round(actor.location.y, 2)
        locationZ = round(actor.location.z, 2)
        bboxW = 2 * round(actor.extent.y, 2)
        bboxL = 2 * round(actor.extent.x, 2)
        bboxH = 2 * round(actor.extent.z, 2)
        rotationYaw = round(actor.rotation.yaw, 2)

        inRangeFlag = isInRange(locationX, locationY)
        # math.sqrt(np.square(locationX - CFG.Lidar['LocationX']) + np.square(locationY - CFG.Lidar['LocationY']))
        if inRangeFlag:
            i += 1
            print("pedestrian:", i, bboxH, bboxW, bboxL, locationX, locationY, locationZ, rotationYaw)

    for actor in vehicle_bbox_list:
        # extent.x extent.y extent.z
        # location.x location.y location.z
        # rotation.pitch rotation.yaw rotation.roll
        locationX = round(actor.location.x, 2)
        locationY = round(actor.location.y, 2)
        locationZ = round(actor.location.z, 2)
        bboxW = 2 * round(actor.extent.y, 2)
        bboxL = 2 * round(actor.extent.x, 2)
        bboxH = 2 * round(actor.extent.z, 2)
        rotationYaw = round(actor.rotation.yaw, 2)

        inRangeFlag = isInRange(locationX, locationY)
        # dis2sensor = math.sqrt(np.square(locationX - CFG.Lidar['LocationX']) + np.square(locationY - CFG.Lidar['LocationY']))
        if inRangeFlag:
            j += 1
            print("vehicle:", j, bboxH, bboxW, bboxL, locationX, locationY, locationZ, rotationYaw)

        label_frame_path = LABEL_PATH.format(point_cloud.frame)

        print(label_frame_path)
        # save_label_data(label_frame_path, label_data)

def isInRange(actor):
    locationX = actor.location.x
    locationY = actor.location.y
    minX = CFG.Lidar_label_range['minX']
    maxX = CFG.Lidar_label_range['maxX']
    minY = CFG.Lidar_label_range['minY']
    maxY = CFG.Lidar_label_range['maxY']
    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag

def creat_kitti_dataset(point_cloud, point_list, world, frame, lidar):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

    carla_world = world
    vehicle_bbox_list = carla_world.get_level_bbs(actor_type=carla.libcarla.CityObjectLabel.Vehicles)
    pedestrians_bbox_list = carla_world.get_level_bbs(actor_type=carla.libcarla.CityObjectLabel.Pedestrians)
    # print("vehicle_bbox_list", vehicle_bbox_list)
    i = 0
    j = 0
    label_data = []
    for actor in vehicle_bbox_list:
        # extent.x extent.y extent.z
        # location.x location.y location.z
        # rotation.pitch rotation.yaw rotation.roll

        if isInRange(actor):
            i += 1
            kitti_datapoint = creat_kitti_datapoint(actor, lidar, 'Car')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    for actor in pedestrians_bbox_list:
        if isInRange(actor):
            j += 1
            kitti_datapoint = creat_kitti_datapoint(actor, lidar, 'Pedestrian')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    # print('label_data', label_data)
    # print('frame: ', frame)
    label_filename = LABEL_PATH.format(frame)
    calib_filename = CALIBRATION_PATH.format(frame)
    if LIDAR_DATA_FORMAT == "bin":
        lidar_filename = LIDAR_PATH.format(frame)
    else:
        lidar_filename = LIDAR_PATH_PLY.format(frame)
    # print('label_filename', )
    save_kitti_label_data(label_filename, label_data)
    save_calibration_matrices(calib_filename)
    save_index_data(INDEX_PATH, frame)
    save_lidar_data(lidar_filename, point_cloud, LIDAR_DATA_FORMAT)
    world.tick()


def dis2sensor(actor):
    dis = math.sqrt(np.square(actor.location.x - CFG.Lidar['LocationX']) + np.square(actor.location.y - CFG.Lidar['LocationY']))
    return dis


def creat_kitti_datapoint(actor, sensor, actor_type):
    if actor:
        datapoint = KittiDescriptor()
        bbox_2d = [0, 0, 0, 0] # no camera so far
        ext = actor.extent
        actor_location = actor.location
        rotation_y = get_relative_rotation_y(actor, sensor)

        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(actor_type)
        datapoint.set_3d_object_location(actor_location)
        datapoint.set_rotation_y(rotation_y)
        return datapoint
    else:
        return None


def get_relative_rotation_y(actor, sensor):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""

    rot_actor = degrees_to_radians(actor.rotation.yaw) # -180 ~ 180 wst clockwise
    # rot_actor = -1 * rot_actor # 180 ~ -180 wst clockwise
    rot_sensor = degrees_to_radians(sensor.get_transform().rotation.yaw)
    rot_y_lidar = rot_actor - rot_sensor # rt wrt lidar coordinate
    rot_y_camera = rot_y_lidar - 0.5*pi
    if rot_y_camera < -1*pi:
        rot_y_camera = rot_y_camera + 2 * pi
    print('vehicle rotation :{}'.format(rot_y_camera))
    # the difference of the x-axis direction between carla and kitti
    kitti_ry = round(rot_y_camera, 2)

    return kitti_ry


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def save_kitti_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti label data to %s", filename)


def proj_to_camera(pos_vector):
    # transform the points to camera
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.

    transformed_3d_pos = np.dot(TR_velodyne, pos_vector)
    return transformed_3d_pos


def save_lidar_data(filename, lidar_measurement, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.

        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        data = np.copy(np.frombuffer(lidar_measurement.raw_data, dtype=np.float32))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        data[:, 1] = -1 * data[:, 1]
        lidar_array = np.array(data).astype(np.float32)
        logging.debug("Lidar min/max of x: {} {}".format(
                      lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of y: {} {}".format(
                      lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of z: {} {}".format(
                      lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)
    else:
        lidar_measurement.save_to_disk(filename)


def save_calibration_matrices(filename):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    """
        in this carla dataset the calibration matrices are set as follows
        3x4 P0: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P1: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P2: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P3: 1 0 0 0 0 1 0 0 0 0 1 0
        3x3 R0_rect: 1 0 0 0 1 0 0 0 1
        3x4 Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = np.identity(3)
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3)
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)


def save_index_data(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        logging.info("Wrote reference files to %s", path)

def main(arg):
    """Main function of the script"""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(5.0)
    world = client.get_world()

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        # vehicle_bp = blueprint_library.filter(arg.filter)[0]
        # vehicle_transform = random.choice(world.get_map().get_spawn_points())
        # vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        # vehicle.set_autopilot(arg.no_autopilot)

        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)

        user_offset = carla.Location(arg.x, arg.y, arg.z)
        # lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        lidar_transform = carla.Transform(carla.Location(x=CFG.Lidar['LocationX'], y=CFG.Lidar['LocationY'], z=CFG.Lidar['LocationZ']),
                                          carla.Rotation(pitch=CFG.Lidar['Pitch'], yaw=CFG.Lidar['Yaw'], roll=CFG.Lidar['Roll']))

        lidar = world.spawn_actor(lidar_bp, lidar_transform)
        # lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        point_list = o3d.geometry.PointCloud()



        if arg.semantic:
            lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
        else:
            # lidar.listen(lambda data: lidar_callback(data, point_list))
            lidar.listen(lambda data: creat_kitti_dataset(data, point_list, world, data.frame, lidar))

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(
        #     window_name='Carla Lidar',
        #     width=960,
        #     height=540,
        #     left=480,
        #     top=270)
        # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        # vis.get_render_option().point_size = 1
        # vis.get_render_option().show_coordinate_frame = True
        #
        # if arg.show_axis:
        #     add_open3d_axis(vis)

        frame = 0
        dt0 = datetime.now()
        while True:
            # if frame == 2:
            #     vis.add_geometry(point_list)
            # vis.update_geometry(point_list)
            #
            # vis.poll_events()
            # vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(delta)


            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            # frame += 1

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        # vehicle.destroy()
        lidar.destroy()
        # vis.destroy_window()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=2.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-24.9,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')