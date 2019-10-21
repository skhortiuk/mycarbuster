#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js]
        [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)]
        [--camera=(single|stereo)] [--meta=<key:value> ...]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...]
        (--model=<model>) [--transfer=<model>]
        [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)]
        [--continuous] [--aug]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    -f --file=<file>   A text file containing paths to tub files, one per line.
                       Option may be used more than once.
    --meta=<key:value> Key/Value strings describing describing a piece of meta
                       data about this drive. Option may be used more than
                       once.
"""

from docopt import docopt

import donkeycar as dk

from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, \
    JoystickController, get_js_controller, JoyStickSub
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.utils import *
from tensorflow.python import keras

from car.camera import CameraConfiguration
from car.controller import ControllerConfiguration


def drive(cfg, model_path: str = None, use_joystick=False, model_type=None,
          camera_type='single', meta="linear"):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a
    job in the Vehicle loop, calling either it's run or run_threaded method
    depending on the constructor flag `threaded`. All parts are updated one
    after another at the framerate given in cfg.DRIVE_LOOP_HZ assuming each
    part finishes processing in a timely manner. Parts may have named
    outputs and inputs. The framework handles passing named outputs to parts
    requesting the same named input.
    """
    print("[*] Started configure vehicle...")

    # Initialize car
    vehicle = dk.vehicle.Vehicle()

    print("[*] Setup camera...")
    CameraConfiguration(vehicle, camera_type).configure_camera()
    print("[*] Camera is ready...")

    print("[*] Setup joystick...")
    controller = ControllerConfiguration(vehicle)
    print("[*] Joystick is ready...")

    # this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    vehicle.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            return mode != "user"

    vehicle.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print(f"[I] Recorded {num_records} records")

            return 0

    rec_tracker_part = RecordTracker()
    vehicle.add(rec_tracker_part, inputs=["tub/num_records"],
                outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        # then we are not using the circle button. hijack that to force a
        # record count indication
        def show_record_amount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1

        # Todo: move to controller
        ctr.set_button_down_trigger('circle', show_record_amount_status)

    class ImgPreProcess:
        """
        preprocess camera image for inference.
        normalize and crop if needed.
        """

        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, img_arr):
            return normalize_and_crop(img_arr, self.cfg)

    inf_input = 'cam/normalized/cropped'
    vehicle.add(ImgPreProcess(cfg),
                inputs=['cam/image_array'],
                outputs=[inf_input],
                run_condition='run_pilot')
    inputs = [inf_input]

    def load_model(kl, model_path):
        print(f"[*] Starting loading model from {model_path}")
        start = time.time()
        kl.load(model_path)
        print(f"[*] Model loaded in {time.time() - start} sec.")

    def load_weights(kl, weights_path):
        try:
            print(f"[*] Starting loading model weights from {weights_path}")
            start = time.time()
            kl.model.load_weights(weights_path)
            print(f"[*] Model weights loaded in {time.time() - start} sec.")
        except Exception as e:
            print(e)
            print(f"[!] Problems loading weights from {weights_path}")

    def load_json_model(kl, json_filename):
        print(f"[*] Starting loading model json from {json_filename}")
        try:
            start = time.time()
            with open(json_filename, 'r') as handle:
                kl.model = keras.models.model_from_json(handle.read())
            print(f"[*] Model loaded in {time.time() - start} sec.")
        except Exception as e:
            print(e)
            print(f"[!] Problems loading model json from {json_filename}")

    if model_path:
        # When we have a model, first create an appropriate Keras part
        keras_model = dk.utils.get_model_by_type(model_type, cfg)
        if not model_path.endswith(".h5"):
            raise Exception(
                "[!] Unknown extension type on model file! Use `.h5`"
            )

        # when we have a .h5 extension
        # load everything from the model file
        load_model(keras_model, model_path)

        def reload_model(filename):
            load_model(keras_model, filename)

        model_reload_cb = reload_model

        # this part will signal visual LED, if connected
        vehicle.add(FileWatcher(model_path, verbose=True),
                    outputs=['modelfile/modified'])

        # these parts will reload the model file, but only when ai is
        # running so we don't interrupt user driving
        vehicle.add(FileWatcher(model_path), outputs=['modelfile/dirty'],
                    run_condition="ai_running")
        vehicle.add(DelayedTrigger(100), inputs=['modelfile/dirty'],
                    outputs=['modelfile/reload'], run_condition="ai_running")
        vehicle.add(TriggeredCallback(model_path, model_reload_cb),
                    inputs=["modelfile/reload"], run_condition="ai_running")

        outputs = ['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        vehicle.add(keras_model, inputs=inputs,
                    outputs=outputs,
                    run_condition='run_pilot')

        # Choose what inputs should change the car.

    class DriveMode:
        def run(self, mode,
                user_angle, user_throttle,
                pilot_angle, pilot_throttle):
            if mode == 'user':
                return user_angle, user_throttle

            elif mode == 'local_angle':
                return pilot_angle, user_throttle

            else:
                return pilot_angle, pilot_throttle * cfg.AI_THROTTLE_MULT

    vehicle.add(DriveMode(),
                inputs=['user/mode', 'user/angle', 'user/throttle',
                        'pilot/angle', 'pilot/throttle'],
                outputs=['angle', 'throttle'])

    # to give the car a boost when starting ai mode in a race.
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE,
                          cfg.AI_LAUNCH_KEEP_ENABLED)

    vehicle.add(aiLauncher,
                inputs=['user/mode', 'throttle'],
                outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON,
                                    aiLauncher.enable_ai_launch)

    class AiRunCondition:
        """
        A bool part to let us know when ai is running.
        """

        def run(self, mode):
            return mode != "user"

    vehicle.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    # Ai Recording
    class AiRecordingCondition:
        """
        return True when ai mode, otherwize respect user mode recording flag
        """

        def run(self, mode, recording):
            return recording if mode == 'user' else True

    if cfg.RECORD_DURING_AI:
        vehicle.add(
            AiRecordingCondition(),
            inputs=['user/mode', 'recording'],
            outputs=['recording']
        )

    # Drive train setup
    if cfg.DONKEY_GYM:
        pass

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=cfg.THROTTLE_FORWARD_PWM,
                               zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                               min_pulse=cfg.THROTTLE_REVERSE_PWM)

        vehicle.add(steering, inputs=['angle'])
        vehicle.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

        steering = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT,
                                             cfg.HBRIDGE_PIN_RIGHT)
        throttle = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD,
                                             cfg.HBRIDGE_PIN_BWD)

        vehicle.add(steering, inputs=['angle'])
        vehicle.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, \
            Mini_HBridge_DC_Motor_PWM

        left_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT_FWD,
                                               cfg.HBRIDGE_PIN_LEFT_BWD)
        right_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_RIGHT_FWD,
                                                cfg.HBRIDGE_PIN_RIGHT_BWD)
        two_wheel_control = TwoWheelSteeringThrottle()

        vehicle.add(two_wheel_control,
                    inputs=['throttle', 'angle'],
                    outputs=['left_motor_speed', 'right_motor_speed'])

        vehicle.add(left_motor, inputs=['left_motor_speed'])
        vehicle.add(right_motor, inputs=['right_motor_speed'])

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
        from donkeycar.parts.actuator import ServoBlaster, PWMSteering
        steering_controller = ServoBlaster(cfg.STEERING_CHANNEL)  # really pin
        # PWM pulse values should be in the range of 100 to 200
        assert (cfg.STEERING_LEFT_PWM <= 200)
        assert (cfg.STEERING_RIGHT_PWM <= 200)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
        motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD,
                                          cfg.HBRIDGE_PIN_BWD)

        vehicle.add(steering, inputs=['angle'])
        vehicle.add(motor, inputs=["throttle"])

    # add tub to save data

    inputs = ['cam/image_array',
              'user/angle', 'user/throttle',
              'user/mode']

    types = ['image_array',
             'float', 'float',
             'str']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label',
                   "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.HAVE_IMU:
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                   'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types += ['float', 'float', 'float',
                  'float', 'float', 'float']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    vehicle.add(tub, inputs=inputs, outputs=["tub/num_records"],
                run_condition='recording')

    if type(ctr) is LocalWebController:
        print("You can now go to <your pi ip address>:8887 to drive your car.")
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        # tell the controller about the tub
        ctr.set_tub(tub)

        if cfg.BUTTON_PRESS_NEW_TUB:
            def new_tub_dir():
                vehicle.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types,
                                        user_meta=meta)
                vehicle.add(tub, inputs=inputs, outputs=["tub/num_records"],
                            run_condition='recording')
                ctr.set_tub(tub)

            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()

    # run the vehicle for 20 seconds
    vehicle.start(rate_hz=cfg.DRIVE_LOOP_HZ,
                  max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']

        dirs = preprocessFileList(args['--file'])
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend(tub_paths)

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)
