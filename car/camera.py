from donkeycar.parts.camera import Webcam, PiCamera, CSICamera, V4LCamera, \
    MockCamera
from donkeycar.parts.image import StereoPair
from donkeycar.parts.cv import CvCam

import config as cfg

from donkeycar import Vehicle


class CameraConfiguration:
    def __init__(self, vehicle: Vehicle, cam_type: str = "single"):
        self._vehicle = vehicle
        self._cam_type = cam_type

    def configure_camera(self):
        print(f"[*] Camera type = {cfg.CAMERA_TYPE}")
        if self._cam_type == "stereo":
            self._configure_stereo_camera()
        else:
            self._configure_single_camera()

    def _configure_stereo_camera(self):
        if cfg.CAMERA_TYPE == "WEBCAM":
            cam_a = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                           image_d=cfg.IMAGE_DEPTH, iCam=0)
            cam_b = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                           image_d=cfg.IMAGE_DEPTH, iCam=1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            cam_a = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                          image_d=cfg.IMAGE_DEPTH, iCam=0)
            cam_b = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                          image_d=cfg.IMAGE_DEPTH, iCam=1)
        else:
            raise Exception(f"Unsupported camera type: {cfg.CAMERA_TYPE}")

        self._vehicle.add(cam_a, outputs=['cam/image_array_a'], threaded=True)
        self._vehicle.add(cam_b, outputs=['cam/image_array_b'], threaded=True)
        self._vehicle.add(
            StereoPair(),
            inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array']
        )

    def _configure_single_camera(self):
        if cfg.CAMERA_TYPE == "PICAM":
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                           image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                         image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                        image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                            image_d=cfg.IMAGE_DEPTH,
                            framerate=cfg.CAMERA_FRAMERATE,
                            gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
        elif cfg.CAMERA_TYPE == "V4L":
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                            image_d=cfg.IMAGE_DEPTH,
                            framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                             image_d=cfg.IMAGE_DEPTH)
        else:
            raise Exception(f"Unknown camera type: {cfg.CAMERA_TYPE}")

        self._vehicle.add(cam, inputs=[], outputs=['cam/image_array'],
                          threaded=True)
