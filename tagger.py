import cv2


def tag_image_points(image, num_points):
    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 4, (0, 255, 0), 2)
            cv2.imshow("image", image)

    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)
    # keep looping until the 'q' key is pressed
    while len(points) < 4:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            points = []
    # close all open windows
    cv2.destroyAllWindows()
    return points
    

if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\avivh\Pictures\vlcsnap-2021-09-16-14h33m12s424.png")
    img = cv2.rotate(img, cv2.ROTATE_180)
    points = tag_image_points(img, 4)
    print(points)
