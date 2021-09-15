#!/usr/bin/python3
from gpiozero import LED, PWMLED
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import copy
#from pid_jataruku import PID
from PID import PID

freq = 25000
mot_l_pins = [PWMLED(19, frequency=freq), PWMLED(26, frequency=freq)] # [backward pin, forward pin]
mot_r_pins = [PWMLED(21, frequency=freq), PWMLED(20, frequency=freq)] # [backward pin, forward pin]

green_led = LED(4)
red_led = LED(5)

def move_wheel(wheel_l_r: str, direction: int, value: float):
    assert wheel_l_r in ['l', 'r']
    motor_pins = mot_r_pins if wheel_l_r == 'r' else mot_l_pins
    if direction == 1:
        motor_pins[direction].value = value
        motor_pins[~direction].off()
    elif direction == -1:
        motor_pins[~direction].value = value
        motor_pins[direction].off()
    else:
        motor_pins[direction].off()
        motor_pins[~direction].off()

def pulse(apin, value=0.2):
    apin.on()
    apin.value = value
    time.sleep(1)
    apin.value = 0
    apin.off()
    
def detect_laser(img):
    ggg = img[:,:,1]
    ind = np.argmax(ggg, axis=None)
    (y, x) = np.unravel_index(ind, ggg.shape)
    val = ggg[y, x]
    return (x, y), val

# Init camera
camera = PiCamera()
frame_size = (320, 240)
camera.resolution = frame_size
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=frame_size)


DISPLAY_FRAMES_FREQ = 0 # Display every N frames (0 to disable)


FLIP_CAMERA = True
RECORD = False
MOTORS_ON = True
LASER_INTENSITY_THRESH = 250
SPEED = 0.6


dx = 0
rotation_out = 0

def set_out(out):
    rotation_out = out


pid = PID(P=0.6, I=0.0, D=0.03, current_time=None)


def plot_values(img, x, y, val):

    cv2.drawMarker(img, (int(x), int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2

    cv2.putText(img, str(val), (100, 100), font, fontScale, 
            color, thickness, cv2.LINE_AA, False)


rec_file = 'vid.raw'
if RECORD:
    # camera.start_recording(rec_file, 'mjpeg')
    # fps = int(camera.framerate)
    # fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
    # writer = cv2.VideoWriter(rec_file, fourcc, fps, frame_size)
    outf = open(rec_file, 'wb')


start = time.time()
iFrame = 0
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    img = copy.copy(frame.array)
    if RECORD:
        cv2.imwrite(f'frame_{iFrame:03d}.bmp', img)
        # outf.write(img.tobytes())
        #  writer.write(img)

    if FLIP_CAMERA:
        img = cv2.rotate(img, cv2.ROTATE_180)

    (x, y), val = detect_laser(img)

    if val > LASER_INTENSITY_THRESH:


        dx = 2 * float(x) / frame_size[0] - 1.0

        pid.update(dx)
        rotation_out = pid.output
        print(f'dx={dx:.2f} out = {rotation_out:.2f}')

        s = min(abs(rotation_out), 1.0)
        if rotation_out < 0:
            # Turn right
            green_led.on()
            red_led.off()
            if MOTORS_ON:
                move_wheel('r', -1, s)
                move_wheel('l', 1, s)
        else:
            # Turn left
            green_led.off()
            red_led.on()
            if MOTORS_ON:
                move_wheel('r', 1, s)
                move_wheel('l', -1, s)

    else:
        green_led.off()
        red_led.off()
        if MOTORS_ON:
            move_wheel('r', 0, 0)
            move_wheel('l', 0, 0)

    if DISPLAY_FRAMES_FREQ > 0 and iFrame % DISPLAY_FRAMES_FREQ == 0:
        plot_values(img, x, y, val)
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    iFrame += 1

    now = time.time()
    elapsed = now - start
    # print(f'Elapsed [ms]: {elapsed*1000.0:.1f}')
    start = now

    W = 100
    
    # xstr = [' '] * W
    # xstr[W//2] = '|'
    # xstr[ int(float(x) / frame_size[0] * W) ] = 'x'
    # sys.stdout.write('\r' + ''.join(xstr))
    # sys.stdout.flush()


    if RECORD and iFrame >= 500:
        break


