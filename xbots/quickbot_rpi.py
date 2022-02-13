"""
@brief QuickBot class

@author Rowland O'Flaherty (rowlandoflaherty.com)
@date 05/30/2014
@version: 2.0
@copyright: Copyright (C) 2014, see the LICENSE file
"""
import os
import sys
import inspect



import threading
import time
from . import base
from . import utils
import re
import serial
import math

#from smbus2 import SMBus, i2c_msg

import numpy as np
from numpy import pi as PI

from . import quickbot_rpi_config as config

from adafruit_motorkit import MotorKit

import py_rpi_hats.tofhat as tofhat

from . base import RIGHT
from . base import LEFT



class QuickBot_rpi(base.BaseBot):
    """The QuickBot Class"""

    # Parameters
    sample_time = 20.0 / 1000.0

    # Motor Pins -- (LEFT, RIGHT)
   # dir1Pin = (config.MOTOR_L.dir1, config.MOTOR_R.dir1)
    #dir2Pin = (config.MOTOR_L.dir2, config.MOTOR_R.dir2)
   # pwmPin = (config.MOTOR_L.pwm, config.MOTOR_R.pwm)

    # LED pin
    #led = config.LED

    # Encoder Serial
 #  encoderSerial = serial.Serial(port=config.ENCODER_SERIAL.port, baudrate=config.ENCODER_SERIAL.baudrate, timeout=.1)
    encoderBuffer = ''

    distance_array = tofhat.tofhat()

    pwm_board = MotorKit()

    # Wheel parameters
    ticksPerTurn = config.encoder_count_per_turn 
    wheelRadius = config.wheel_radius_in_meters

    # State encoder
    enc_raw = [0.0, 0.0]  # Last encoder tick position
    enc_vel = [0.0, 0.0]  # Last encoder tick velocity
    enc_est_vel = [0.0,  0.0] # estimate of encoder tick velocity
    enc_time = 0.0  # Last encoder tick sample time
    enc_state = 2*[np.array([[0.0, 0.0]]).T]  # Last encoder state -- pos, vel, acc
    enc_state_cov = 2*[np.array([[1.0, 0.0],
                      [0.0, 1.0]])]

    # Controller paramerters
    enc_vel_controller_flag = 2*[False]

    def __init__(self, base_ip, robot_ip):
        super(QuickBot_rpi, self).__init__(base_ip, robot_ip)


        self.i2c_lock = threading.Lock()

        # Start time
        self.t0 = time.time()

        # State IR
        self.n_ir = len(self.distance_array.sensors)
        self.ir_val = self.n_ir*[0.0]

        # State Encoder
        self.enc_dir = [1, -1]         # Last encoder direction
        self.enc_raw = [0, 0]          # Last encoder tick position
        self.enc_vel = [0.0, 0.0]      # Last encoder tick velocity
        self.enc_offset = [0.0, 0.0]   # Offset from raw encoder tick

        # Set Points
        self.enc_vel_set_point = [0.0, 0.0]

          # Initialize IR thread
        self.ir_thread = threading.Thread(target=read_ir_thread_fcn, args=(self, ))
        self.ir_thread.daemon = True

        # Initialize encoder thread
        self.enc_pos_thread = threading.Thread(
                target=read_enc_val_thread_fcn, args=(self,))
        self.enc_pos_thread.daemon = True

        # Initialize wheel controller thread
        self.enc_vel_controller_thread = 2*[None]
        for side in range(0, 2):
            self.enc_vel_controller_thread[side] = threading.Thread(
                target=enc_vel_controller_thread_fcn, args=(self, side))
            self.enc_vel_controller_thread[side].daemon = True


    def start_threads(self):
        """ Start all threads """
        self.ir_thread.start()
        self.enc_pos_thread.start()
        for side in range(0, 2):
            self.enc_vel_controller_thread[side].start()

        # Calibrate encoders
        self.calibrate_enc_val()

        # Call parent method
        super(QuickBot_rpi, self).start_threads()

    def get_pwm(self):
        """ Get motor PWM values """
        self.i2c_lock.acquire()
        try:
            left = self.pwm_board.motor1.throttle*100
            right = self.pwm_board.motor3.throttle*100
        except Exception as e:
            print("Exception: " + str(e))
        finally:
            self.i2c_lock.release()
        return [left,right]

    def set_pwm(self, values):
        self.set_pwm_left(values[0])
        self.set_pwm_right(values[1])

    def set_pwm_left(self, pwm):
        """ Set left motor PWM value """
        self.i2c_lock.acquire()
        try:
            self.enc_vel_controller_flag[LEFT] = False
            self.pwm_board.motor1.throttle = pwm/100.0
        except Exception as e:
            print("Exception: " + str(e))
        finally:
            self.i2c_lock.release()            
        
    def set_pwm_right(self, pwm):
        """ Set right motor PWM value """
        self.i2c_lock.acquire()
        try:
            self.enc_vel_controller_flag[RIGHT] = False
            self.pwm_board.motor3.throttle = pwm/100.0
        except Exception as e:
            print("Exception: " + str(e))
        finally:
            self.i2c_lock.release()

    def get_ir(self):
        """ Getter for IR sensor values """
        return self.ir_val

    def calibrate_enc_val(self):
        """ Calibrate wheel encoder values"""
        self.set_pwm([100, 100])
        time.sleep(0.1)
        self.set_pwm([0, 0])
        time.sleep(1.0)
        self.reset_enc_val()

    def get_enc_raw(self):
        """ Getter for raw encoder values """
        return self.enc_raw

    def get_enc_val(self):
        """ Getter for encoder tick values i.e (raw - offset) """
        return [self.enc_raw[LEFT] - self.enc_offset[LEFT],
                -1*(self.enc_raw[RIGHT] - self.enc_offset[RIGHT])]

    def set_enc_val(self, enc_val):
        """ Setter for encoder tick positions """
        offset = [0.0, 0.0]
        offset[LEFT] = self.enc_raw[LEFT] - enc_val[LEFT]
        offset[RIGHT] = (self.enc_raw[RIGHT] - enc_val[RIGHT])
        self.set_enc_offset(offset)

#    def get_wheel_ang(self):
#        """ Getter for wheel angles """
#        ang = [0.0, 0.0]
#        enc_val = self.get_enc_val()
#        for side in range(0, 2):
#            ang[side] = enc_val[side] / self.ticksPerTurn * 2 * PI
#        return ang

#    def set_wheel_ang(self, ang):  # FIXME - Should move wheel to that angle
#        """ Setter for wheel angles """
#        enc_val = [0.0, 0.0]
#        for side in range(0, 2):
#            enc_val[side] = ang[side] * self.ticksPerTurn / (2 * PI)
#        self.set_enc_val(enc_val)

    def get_enc_offset(self):
        """ Getter for encoder offset values """
        return self.enc_offset

    def set_enc_offset(self, offset):
        """ Setter for encoder offset values """
        for side in range(0, 2):
            self.enc_offset[side] = offset[side]

    def reset_enc_val(self):
        """ Reset encoder values to 0 """
        self.enc_offset[LEFT] = self.enc_raw[LEFT]
        self.enc_offset[RIGHT] = self.enc_raw[RIGHT]

    def get_enc_vel(self):
        """ Getter for encoder velocity values """
        return self.enc_vel
    
    def get_enc_est_vel(self):
        return self.enc_est_vel

    def set_enc_vel(self, env_vel):
        """ Setter for encoder velocity values """
        for side in range(0, 2):
            self.enc_vel_set_point[side] = env_vel[side]
        self.enc_vel_controller_flag = 2*[True]

#    def get_wheel_ang_vel(self):
#        """ Getter for wheel angular velocity values """
#        ang_vel = [0.0, 0.0]
#        enc_vel = self.get_enc_vel()
#        for side in range(0, 2):
#            ang_vel[side] = enc_vel[side] * (2 * PI) / self.ticksPerTurn
#        return ang_vel

#    def set_wheel_ang_vel(self, ang_vel):
#        """ Setter for wheel angular velocity values """
#        for side in range(0, 2):
#            self.enc_vel_set_point[side] = ang_vel[side] * self.ticksPerTurn / (2 * PI)
#        self.enc_vel_controller_flag = 2*[True]


def enc_vel_controller_thread_fcn(self, side):
    """ Thread function for controlling for encoder tick velocity """
    while self.run_flag:
        if self.enc_vel_controller_flag[side]:
            x = self.enc_vel[side]
            u = self.pwm[side]
            x_bar = self.enc_vel_set_point[side]

            u_plus = enc_vel_controller(x, u, x_bar)

            if side == LEFT:
                self.set_pwm_left(u_plus)
            else:
                self.set_pwm_right(u_plus)

        time.sleep(self.sample_time)

def enc_vel_controller(x, u, x_bar):
    """ Wheel angular velocity controller """
    controller_type = 'PID'

    if controller_type == 'PID':
        P = 0.05
        u_plus = P * (x_bar - x) + u

    return u_plus


def read_ir_thread_fcn(self):
    """ Thread function for reading IR sensor values """
    while self.run_flag:
        index = 0
        for s in self.distance_array.sensors:
            self.i2c_lock.acquire()
            try:
                measurement = s.sensor.range
            except Exception as e:
                print("Exception: " + str(e))
            finally:
                self.i2c_lock.release()

            if measurement != -1:
                 self.ir_val[index] = measurement
            index += 1
        time.sleep(0.050)
        


def read_enc_val_thread_fcn(self):
    """ Thread function for reading encoder values """
    with serial.Serial(port=config.ENCODER_SERIAL.port, baudrate=config.ENCODER_SERIAL.baudrate, timeout=.1) as self.encoderSerial:
        while self.run_flag:
            parse_encoder_buffer(self)
            time.sleep(0.1)


def parse_encoder_buffer(self):
    """ Parses encoder serial data """

    t = time.time() - self.t0
    ts = t - self.enc_time

    zl = np.array([[np.NaN]])
    zr = np.array([[np.NaN]])

    bytes_in_waiting = self.encoderSerial.inWaiting()

    if bytes_in_waiting > 0:
        try:
            self.encoderBuffer += self.encoderSerial.read(bytes_in_waiting).decode("utf-8") 
        except(SerialException) as e:
            return


        if len(self.encoderBuffer) > 84:
            self.encoderBuffer = self.encoderBuffer[-42:]

        if len(self.encoderBuffer) >= 42:
            d_pattern = r'LD:([0-9A-Fa-f]{8})'
            d_regex = re.compile(d_pattern)
            d_result = d_regex.findall(self.encoderBuffer)
            if len(d_result) >= 1:
                val = utils.convertHEXtoDEC(d_result[-1], 8)
                if not math.isnan(val):
                    self.enc_raw[0] = val
                    enc_pos_update_flag = True
            
            v_pattern = r'LV:([0-9A-Fa-f]{4})'
            v_regex = re.compile(v_pattern)
            v_result = v_regex.findall(self.encoderBuffer)
            if len(v_result) >= 1:
                vel = utils.convertHEXtoDEC(v_result[-1], 4)
                if not math.isnan(vel):
                    self.enc_vel[0] = vel
                    zl = np.array([[vel]])

            d_pattern = r'RD:([0-9A-Fa-f]{8})'
            d_regex = re.compile(d_pattern)
            d_result = d_regex.findall(self.encoderBuffer)
            if len(d_result) >= 1:
                val = utils.convertHEXtoDEC(d_result[-1], 8)
                if not math.isnan(val):
                    self.enc_raw[1] = val
                    enc_pos_update_flag = True
            
            v_pattern = r'RV:([0-9A-Fa-f]{4})'
            v_regex = re.compile(v_pattern)
            v_result = v_regex.findall(self.encoderBuffer)
            if len(v_result) >= 1:
                vel = utils.convertHEXtoDEC(v_result[-1], 4)
                if not math.isnan(vel):
                    self.enc_vel[1] = vel
                    zr = np.array([[vel]])


    ul = self.pwm[0]
    xl = self.enc_state[0]
    Pl = self.enc_state_cov[0]
  

    ur = self.pwm[1]
    xr = self.enc_state[1]
    Pr = self.enc_state_cov[1]

    A = np.array([[1.0,  ts],
                  [0.0, 0.0]])
    B = np.array([[6.0]])
    C = np.array([[0.0, 1.0]])
    W = np.array([1.0, 1.0])
    V = np.array([0.5])

    (x_p, P_p) = utils.kalman(xl, ul, Pl, A, B, C, W, V, zl)

    self.enc_state[0] = x_p
    self.enc_state_cov[0] = P_p

    self.enc_time = t
    self.enc_est_vel[0] = np.asscalar(x_p[[1]])

    (x_p, P_p) = utils.kalman(xr, ur, Pr, A, B, C, W, V, zr)
    self.enc_state[1] = x_p
    self.enc_state_cov[1] = P_p

    self.enc_est_vel[1] = np.asscalar(x_p[[1]])

    return
