"""
@brief Base robot class

@author Josip Delic (delijati.net)
@author Rowland O'Flaherty (rowlandoflaherty.com)
@date 04/23/2014
@version: 1.0
@copyright: Copyright (C) 2014, see the LICENSE file
"""

import sys
import time
import re
import socket
import threading

import Adafruit_BBIO.GPIO as GPIO  #pylint: disable=F0401
import Adafruit_BBIO.PWM as PWM  #pylint: disable=F0401

LEFT = 0
RIGHT = 1
MIN = 0
MAX = 1

DEBUG = False


class BaseBot(object):
  """Base robot class. Mainly handles initialization and messages passing."""

  # Parameters
  sampleTime = 20.0 / 1000.0
  pwm_freq = 2000

  # Variables
  ledFlag = True
  cmdBuffer = ''

  # Motor Pins -- (LEFT, RIGHT)
  dir1Pin = ()
  dir2Pin = ()
  pwmPin = ()

  # UDP
  port = 5005
  robotSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  robotSocket.setblocking(False)

  def __init__(self, base_ip, robot_ip):
    # Run flag
    self.run_flag = True

    # Initialize GPIO pins
    self._setup_gpio()

    # Initialize PWM pins: PWM.start(channel, duty, freq=2000, polarity=0)
    self._init_pwm(self.pwm_freq)

    # Set motor speed to 0
    self.setPWM([0, 0])

    # Set IP addresses
    self.base_ip = base_ip
    self.robot_ip = robot_ip
    self.robotSocket.bind((self.robot_ip, self.port))

    # Initialize command parsing thread
    self.cmdParsingThread = threading.Thread(target=parseCmd, args=(self,))
    self.cmdParsingThread.daemon = True

  def _setup_gpio(self):
    """Initialize GPIO pins"""
    GPIO.setup(self.dir1Pin[LEFT], GPIO.OUT)
    GPIO.setup(self.dir2Pin[LEFT], GPIO.OUT)
    GPIO.setup(self.dir1Pin[RIGHT], GPIO.OUT)
    GPIO.setup(self.dir2Pin[RIGHT], GPIO.OUT)
    GPIO.setup(self.led, GPIO.OUT)

  def _init_pwm(self, frequency=2000):
    """ Initialize PWM pins:
      PWM.start(channel, duty, freq=2000, polarity=0)
    """
    # XXX
    # It is currently not possible to set frequency for two PWM
    # a maybe solution patch pwm_test.c
    # https://github.com/SaadAhmad/beaglebone-black-cpp-PWM
    PWM.start(self.pwmPin[LEFT], 0)
    PWM.start(self.pwmPin[RIGHT], 0)

  def setPWM(self, pwm):
    # [leftSpeed, rightSpeed]: 0 is off, caps at min and max values

    self.pwm[LEFT] = min(
      max(pwm[LEFT], self.pwmLimits[MIN]), self.pwmLimits[MAX])
    self.pwm[RIGHT] = min(
      max(pwm[RIGHT], self.pwmLimits[MIN]), self.pwmLimits[MAX])

    # Left motor
    if self.pwm[LEFT] > 0:
      GPIO.output(self.dir1Pin[LEFT], GPIO.LOW)
      GPIO.output(self.dir2Pin[LEFT], GPIO.HIGH)
      PWM.set_duty_cycle(self.pwmPin[LEFT], abs(self.pwm[LEFT]))
    elif self.pwm[LEFT] < 0:
      GPIO.output(self.dir1Pin[LEFT], GPIO.HIGH)
      GPIO.output(self.dir2Pin[LEFT], GPIO.LOW)
      PWM.set_duty_cycle(self.pwmPin[LEFT], abs(self.pwm[LEFT]))
    else:
      GPIO.output(self.dir1Pin[LEFT], GPIO.LOW)
      GPIO.output(self.dir2Pin[LEFT], GPIO.LOW)
      PWM.set_duty_cycle(self.pwmPin[LEFT], 0)

    # Right motor
    if self.pwm[RIGHT] > 0:
      GPIO.output(self.dir1Pin[RIGHT], GPIO.LOW)
      GPIO.output(self.dir2Pin[RIGHT], GPIO.HIGH)
      PWM.set_duty_cycle(self.pwmPin[RIGHT], abs(self.pwm[RIGHT]))
    elif self.pwm[RIGHT] < 0:
      GPIO.output(self.dir1Pin[RIGHT], GPIO.HIGH)
      GPIO.output(self.dir2Pin[RIGHT], GPIO.LOW)
      PWM.set_duty_cycle(self.pwmPin[RIGHT], abs(self.pwm[RIGHT]))
    else:
      GPIO.output(self.dir1Pin[RIGHT], GPIO.LOW)
      GPIO.output(self.dir2Pin[RIGHT], GPIO.LOW)
      PWM.set_duty_cycle(self.pwmPin[RIGHT], 0)

  def run(self):
    # Start threads
    self.cmdParsingThread.start()
    self.startThreads()

    # Run loop
    while self.run_flag:
      self.update()

      # Flash BBB LED
      if self.ledFlag is True:
        self.ledFlag = False
        GPIO.output(self.led, GPIO.HIGH)
      else:
        self.ledFlag = True
        GPIO.output(self.led, GPIO.LOW)

      time.sleep(self.sampleTime)
    self.cleanup()
    return

  def cleanup(self):
    sys.stdout.write("Shutting down...")
    self.setPWM([0, 0])
    self.robotSocket.close()
    GPIO.cleanup()
    PWM.cleanup()
    if DEBUG:
      pass
      # tictocPrint()
      # self.writeBuffersToFile()
    self.waitForThreads()
    sys.stdout.write("Done\n")

  def writeBuffersToFile(self):
    matrix = map(list, zip(*[self.encTimeRec[LEFT], self.encValRec[LEFT],
                 self.encPWMRec[LEFT], self.encNNewRec[LEFT],
                 self.encTickStateRec[LEFT],
                 self.encPosRec[LEFT],
                 self.encVelRec[LEFT],
                 self.encThresholdRec[LEFT],
                 self.encTimeRec[RIGHT],
                 self.encValRec[RIGHT],
                 self.encPWMRec[RIGHT], self.encNNewRec[RIGHT],
                 self.encTickStateRec[RIGHT],
                 self.encPosRec[RIGHT], self.encVelRec[RIGHT],
                 self.encThresholdRec[RIGHT]]))
    s = [[str(e) for e in row] for row in matrix]
    lens = [len(max(col, key=len)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    f = open('output.txt', 'w')
    f.write('\n'.join(table))
    f.close()
    print "Wrote buffer to output.txt"


def parseCmd(self):
  try:
    while self.run_flag:
      try:
        line = self.robotSocket.recv(1024)
      except socket.error as msg:
        continue

      self.cmdBuffer += line

      # String contained within $ and * symbols with no $ or * symbols in it
      bufferPattern = r'\$[^\$\*]*?\*'
      bufferRegex = re.compile(bufferPattern)
      bufferResult = bufferRegex.search(self.cmdBuffer)

      if bufferResult:
        msg = bufferResult.group()
        print msg
        self.cmdBuffer = ''

        msgCmdPattern = '(?P<CMD>[A-Z]{3,})'
        msgSetPattern = '(?P<SET>=?)'
        msgQueryPattern = '(?P<QUERY>\??)'
        msgArgPattern = '(?(2)(?P<ARGS>.*))'
        msgPattern = r'\$' + \
          msgCmdPattern + \
          msgSetPattern + \
          msgQueryPattern + \
          msgArgPattern + \
          '.*\*'

        msgRegex = re.compile(msgPattern)
        msgResult = msgRegex.search(msg)

        if msgResult.group('CMD') == 'CHECK':
          self.robotSocket.sendto(
            'Hello from QuickBot\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'PWM':
          if msgResult.group('QUERY'):
            self.robotSocket.sendto(
              str(self.pwm) + '\n', (self.base_ip, self.port))

          elif msgResult.group('SET') and msgResult.group('ARGS'):
            args = msgResult.group('ARGS')
            pwmArgPattern = r'(?P<LEFT>[-]?\d+),(?P<RIGHT>[-]?\d+)'
            pwmRegex = re.compile(pwmArgPattern)
            pwmResult = pwmRegex.match(args)
            if pwmResult:
              pwm = [int(pwmRegex.match(args).group('LEFT')),
                   int(pwmRegex.match(args).group('RIGHT'))]
              self.setPWM(pwm)

        elif msgResult.group('CMD') == 'IRVAL':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.getIr())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ULTRAVAL':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.ultraVal)) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'POS':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.getPos())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ENPOS' or \
            msgResult.group('CMD') == 'ENVAL':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.getEncPos())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ENRAW':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.getEncRaw())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ENOFFSET':
          if msgResult.group('QUERY'):
            reply = \
              '[' + ', '.join(map(str, self.getEncOffset())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ENVEL':
          if msgResult.group('QUERY'):
            reply = '[' + ', '.join(map(str, self.getEncVel())) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'ENRESET':
          self.resetEncPos()
          print 'Encoder values reset to [' + ', '.join(
            map(str, self.encVel)) + ']'

        elif msgResult.group('CMD') == 'UPDATE':
          if msgResult.group('SET') and msgResult.group('ARGS'):
            args = msgResult.group('ARGS')
            pwmArgPattern = r'(?P<LEFT>[-]?\d+),(?P<RIGHT>[-]?\d+)'
            pwmRegex = re.compile(pwmArgPattern)
            pwmResult = pwmRegex.match(args)
            if pwmResult:
              pwm = [int(pwmRegex.match(args).group('LEFT')),
                   int(pwmRegex.match(args).group('RIGHT'))]
              self.setPWM(pwm)

            reply = '[' + ', '.join(map(str, self.encPos)) + ', ' \
              + ', '.join(map(str, self.encVel)) + ']'
            print 'Sending: ' + reply
            self.robotSocket.sendto(
              reply + '\n', (self.base_ip, self.port))

        elif msgResult.group('CMD') == 'END':
          self.run_flag = False
  except:
    self.run_flag = False
    raise
