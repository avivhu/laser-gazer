# How to obtain calibration images:
# 1. Start streaming output to PC:
#    mjpg_streamer -i input_"raspicam.so -x 320 -y 200 -fps 5" -o output_http.so
# 2. Open stream in vlc 
#    http://raspberrypi:8080/?action=stream
# For more information, see: https://www.okdo.com/project/pc-webcam-with-raspberry-pi/
# 3. Take snapshots. In VLC->Video->Take snapshot

#%%
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


PATTERN_SIZE = (9, 6)
SQUARE_SIZE_CM = 3.4 # Measured from my source checkerboard
ROTATE_CAMERA_180 = True


im_paths = list(Path('./calib_data').glob('*.png'))
ims = [cv2.imread(str(p)) for p in im_paths]


if ROTATE_CAMERA_180:
    ims = [cv2.rotate(im, cv2.ROTATE_180) for im in ims]


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp *=  SQUARE_SIZE_CM



retval = [None] * len(ims)
corners = [None] * len(ims)
corners2 = [None] * len(ims)
display_ims = [None] * len(ims)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i, im in enumerate(ims):
    im = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    retval[i], corners[i] = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    display_ims[i] = cv2.drawChessboardCorners(im, PATTERN_SIZE, corners[i], retval[i])

    if retval[i] == True:
        objpoints.append(objp)
        corners2[i] = cv2.cornerSubPix(gray, corners[i], (11,11), (-1,-1), criteria)
        imgpoints.append(corners2[i])
        # draw_axis and display the corners
        cv2.drawChessboardCorners(im, PATTERN_SIZE, corners2[i], retval[i])
        cv2.imshow('img', im)
        cv2.waitKey(500)
cv2.destroyAllWindows()

#%%
ok, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
assert ok

#%%

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = tuple(int(x) for x in corner)
    imgpts = imgpts.astype(int)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for i, im in enumerate(ims):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if retval[i] == True:
        # Find the rotation and translation vectors.
        objp = objpoints[i]

        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2[i], mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw_axis(im, corners2[i], imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(500)
cv2.destroyAllWindows()
#%%
floor_im = cv2.imread(r"C:\Users\avivh\Pictures\vlcsnap-2021-09-16-14h33m12s424.png")
if ROTATE_CAMERA_180:
    floor_im = cv2.rotate(floor_im, cv2.ROTATE_180)
points = [(158, 127), (240, 121), (292, 144), (173, 151)]


plt.imshow(floor_im)
for pt in points:
    plt.plot(pt[0], pt[1], marker='+')

FLOOR_TILE_SIZE_CM = 20

floor_tile_points = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]) * FLOOR_TILE_SIZE_CM
# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv2.solvePnP(np.array(floor_tile_points).astype(float), np.array(points).astype(float), mtx, dist)

rotM = cv2.Rodrigues(rvecs.flatten())[0]
cameraPosition = -np.matrix(rotM).T @ np.matrix(tvecs.flatten()).T

uvPoint = np.array([0, 0, 1])
zConst = 0
tempMat = np.linalg.inv(rotM) @ np.linalg.inv(mtx) @ uvPoint
tempMat2 = np.linalg.inv(rotM) @ tvecs
s = zConst + tempMat2[2,0]
s /= tempMat[2]
wcPoint = np.linalg.inv(rotM)  @ (s * np.linalg.inv(mtx)  @ uvPoint - tvecs)
ppp = wcPoint / wcPoint.flatten()[-1]
# Plot points between camera and focus tile
minx = cameraPosition


# Draw image around tile


xs = np.arange(-20, 20, 1)
ys = np.arange(-20, 20, 1)
pts = np.meshgrid(xs, ys)




def transform_points(pts, m):
    if pts.shape[1] == 2:
        pts = np.hstack((pts, np.ones((len(pts), 1))))
    assert pts.shape[1] == 3
    res = m @ pts.T
    res = res / res[-1, :]
    return res

# Make a new homography that maps the image center point into 0,0
img_center_pt = np.array((320/2, 150))

img_points = np.array(points).astype(float)
img_points[:, 0] += -img_points[0, 0] + img_center_pt[0]
img_points[:, 1] += -img_points[0, 1] + img_center_pt[1]
homog, _ = cv2.findHomography(img_points, np.array(floor_tile_points).astype(float), )

# img_center_pt_world = transform_points(img_center_pt, homog)
# homog2 = homog.copy()
# homog2[0, 2] -= img_center_pt_world.flatten()[0]
# homog2[1, 2] -= img_center_pt_world.flatten()[1]
# delta = [
#     [1, 0, -img_center_pt_world.flatten()[0]], 
#     [0, 1, -img_center_pt_world.flatten()[1]], 
#     [0, 0, 1]]
# delta @ np.array([[1,0,1]]).T
# homog2  = homog + delta

# homog @ np.array([0, 0, 1]).T

roi_to_render = [-20, -20, 0]

transform_points(np.array([img_center_pt]), homog)
transform_points(np.array([img_center_pt]), homog)

im_dst = cv2.warpPerspective(floor_im, homog, (40, 40))


H, W = floor_im.shape[:2]
img_roi = np.array([[0, H/2], [W, H/2], [W, H], [0, H]])
floor_roi = transform_points(img_roi, homog)
transform_points(np.array([img_center_pt]),  np.linalg.inv(homog))

pts = np.array(points).astype(float)
pts.shape[1]
pts2 = np.hstack((pts, np.ones((len(pts), 1))))
res = homog @ pts2.T
res = res / res[-1, :]

transform_points(aaa, homog)



aaa = np.array(floor_tile_points).astype(float)


aaa = aaa[:, 0:2]
cv2.perspectiveTransform(np.array(floor_tile_points).astype(float), homog)
plt.imshow(im_dst)
plt.show()

a = homog @ np.array([240, 121, 1]).T
a /= a[2]
points

# %%
