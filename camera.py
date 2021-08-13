import pyrealsense2 as rs
import numpy as np
import cv2
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
ptime = 0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rois = []
        roi1 = color_image[0:640,0:160]
        roi2 = color_image[0:640,160:320]
        roi3 = color_image[0:640,320:480]
        rois.append(roi1)
        rois.append(roi2)
        rois.append(roi3)
        
        # roi1 için
        hsvim = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        
        kernel = np.ones((7,7),dtype=np.uint8)
        opening = cv2.morphologyEx(blurred.astype(np.float32),cv2.MORPH_OPEN,kernel)

        count = []
        ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 500
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                count.append(contours[i])
                max_area = area
        max_area1 = max_area
        #print("max area 1: " + str(max_area))
        contours = count
        hull = []
        defects = []
    
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i]))
        
        cv2.drawContours(roi1, hull, -1, (255, 0, 0), 2)
        cv2.drawContours(roi1, contours, -1, (0, 255, 255), 2)

        #roi2 için
        hsvim = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        
        

        count = []
        ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        kernel = np.ones((7,7),dtype=np.uint8)
        opening = cv2.morphologyEx(thresh.astype(np.float32),cv2.MORPH_OPEN,kernel)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 500
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                count.append(contours[i])
                max_area = area
        max_area2 = max_area
        #print("max area 2:  " + str(max_area2))

        contours = count
        hull = []
        defects = []
    
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i]))
        
        cv2.drawContours(roi2, hull, -1, (255, 0, 0), 2)
        cv2.drawContours(roi2, contours, -1, (0, 255, 255), 2)
        
        #roi3 için
        hsvim = cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        
        kernel = np.ones((7,7),dtype=np.uint8)
        opening = cv2.morphologyEx(blurred.astype(np.float32),cv2.MORPH_OPEN,kernel)

        count = []
        ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 700
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                count.append(contours[i])
                max_area = area
        max_area3 = max_area
        #print("max area 3:  " + str(max_area3))

        contours = count
        hull = []
        defects = []
    
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i]))
        
        cv2.drawContours(roi3, hull, -1, (255, 0, 0), 2)
        cv2.drawContours(roi3, contours, -1, (0, 255, 255), 2)
        
        """for i in range(len(contours)):
                defects.append(cv2.convexityDefects(contours[i], hull[i]))"""
        
        """if defects is not None:
                cnt = 0
        for i in range(len(defects)):  # calculate the angle
                print(defects)
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem

        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
            cv2.circle(color_image, far, 4, [0, 0, 255], -1)

        if cnt > 0:
                cnt = cnt+1

        cv2.putText(color_image, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
       """
        if (max_area1 > max_area2) and  (max_area1 > max_area3):

                cv2.putText(color_image,"OK",(70,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255) , 2, cv2.LINE_AA)

        elif (max_area2 > max_area1) and  (max_area2 > max_area3):

                cv2.putText(color_image,"OK",(70,100),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255) , 2, cv2.LINE_AA)
        elif (max_area3 > max_area2) and  (max_area3 > max_area1):

                cv2.putText(color_image,"OK",(70,150),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255) , 2, cv2.LINE_AA)
        cv2.line(color_image,(213,0),(213,480),(255,0,0),2,cv2.LINE_4)
        cv2.line(color_image,(426,0),(426,480),(255,0,0),2,cv2.LINE_4)

        cv2.putText(color_image, "A", (100, 450), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.putText(color_image, "B", (300, 450), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.putText(color_image, "C", (550, 450), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)

        cv2.putText(color_image, "A", (40, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.putText(color_image, "B", (40, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.putText(color_image, "C", (40, 150), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.rectangle(color_image, (70, 20), (120, 60),(0,255,0),2)
        cv2.rectangle(color_image, (70, 70), (120, 110),(0,255,0),2)
        cv2.rectangle(color_image, (70, 120), (120, 160),(0,255,0),2)

        """cv2.imshow("roi1",roi1)
        cv2.imshow("roi2",roi2)
        cv2.imshow("roi3",roi3)"""
        cv2.imshow("full", color_image)
        cv2.waitKey(1)
       
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        """depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)"""

finally:

    # Stop streaming
    pipeline.stop()