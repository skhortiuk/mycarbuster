from donkeycar import Vehicle
from donkeycar.parts.controller import get_js_controller, JoyStickSub
from donkeycar.parts.web_controller.web import LocalWebController

import config as cfg


class ControllerConfiguration:
    def __init__(self, vehicle: Vehicle):
        self._vehicle = vehicle
        self._controller = None

    def configure(self, use_joystick: bool):
        if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
            # modify max_throttle closer to 1.0 to have more power modify
            # steering_scale lower than 1.0 to have less responsive steering
            self._controller = get_js_controller(cfg)

            if cfg.USE_NETWORKED_JS:
                network_js = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                self._vehicle.add(network_js, threaded=True)
                self._controller.js = network_js

        else:
            # This web controller will create a web server that is capable
            # of managing steering, throttle, and modes, and more.
            self._controller = LocalWebController()

        self._vehicle.add(
            self._controller,
            inputs=['cam/image_array'],
            outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
            threaded=True
        )

    @property
    def controller(self):
        return self._controller
