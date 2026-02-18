import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
capture = cv.VideoCapture('volleyball_match.mp4')

def get_team_color(frame, box, red_mask, yellow_mask):
    x1, y1, x2, y2 = box
    
    torso_y1 = y1
    torso_y2 = y1 + int((y2 - y1) * 0.4)
    
    torso_red = red_mask[torso_y1:torso_y2, x1:x2]
    torso_yellow = yellow_mask[torso_y1:torso_y2, x1:x2]
    
    red_pixels = np.sum(torso_red > 0)
    yellow_pixels = np.sum(torso_yellow > 0)
    
    if red_pixels > 0 and red_pixels > yellow_pixels: # changed to zero
        return 'red'
    elif yellow_pixels > 0 and yellow_pixels > red_pixels:
        return 'yellow'
    else:
        return 'unknown'
    

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def add_center(arr, c):
    if len(arr) < 20:
        arr.append(c)
    else:
        for i in range(19):
            arr[i] = arr[i+1]
        arr[19] = c

def mouse_callback(event, x, y, flags, param):
    global hsv
    if event == cv.EVENT_LBUTTONDOWN:
        h, s, v = hsv[y, x]
        print(f"Clicked at ({x}, {y}) -> HSV: [{h}, {s}, {v}]")

cv.namedWindow('Video')
cv.setMouseCallback('Video', mouse_callback)


lower = np.array([0, 80, 100])
upper = np.array([30, 255, 255])

lower_red = np.array([170, 180, 140])
upper_red = np.array([180, 255, 255])

lower_white = np.array([0, 0, 210])
upper_white = np.array([180, 70, 255])

lower_yellow = np.array([8, 160, 150])
upper_yellow = np.array([30, 255, 255])

prev_center = np.array([933, 192])
listofcenters = []
lost_frames = 0
MAX_LOST_FRAMES = 10  


tracked_persons = {}  # Dictionary to track persons across frames
next_person_id = 0


while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    
    blur = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    red_mask1 = cv.inRange(hsv, lower_red, upper_red)
    red_mask2 = cv.inRange(hsv, lower_white, upper_white)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    contours, hierarchies = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < 50 or area > 1000:
            continue
        
        (x, y), r = cv.minEnclosingCircle(cnt)
        if r < 4 or r > 20:
            continue
        
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:
            continue
        
        rx, ry, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.55 or aspect_ratio > 1.45: # ideal 1
            continue
        
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < 0.65: # ideal 1
            continue

        
        candidates.append((cnt, (int(x), int(y)), r, circularity, area))
    
    best = None
    best_score = float('inf')
    
    for cnt, (x, y), r, circularity, area in candidates:
        if prev_center is None:
            score = (1 - circularity) * 100 + abs(r - 30)  # badness
        else:
            dist = np.linalg.norm(np.array([x, y]) - np.array(prev_center))
            max_dist = 150 if lost_frames < 5 else 300
            if dist > max_dist or dist < 1:
                continue
            score = dist * 0.7 + (1 - circularity) * 50 + abs(r - 30) * 2
        
        if score < best_score:
            best_score = score
            best = (cnt, (x, y), r, circularity)

    if best is not None:
        cnt, (x, y), r, circularity = best
        prev_center = [x, y]
        lost_frames = 0
        add_center(listofcenters, prev_center)

        cv.circle(frame, (x, y), int(r), (0, 255, 0), 2)
        cv.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Center point
        for i in range(len(listofcenters)-1):
            cv.line(frame, listofcenters[i], listofcenters[i+1], (0, 255, 255), 1)
        # cv.putText(frame, f"Ball R={r:.1f} C={circularity:.2f}", 
        #            (x - 50, y - int(r) - 10), 
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        lost_frames += 1
        if lost_frames > MAX_LOST_FRAMES:
            prev_center = None  # Reset tracking
    
    # Draw all candidates for debugging
    # for cnt, (x, y), r, circularity, area in candidates:
    #     cv.circle(frame, (x, y), int(r), (255, 0, 0), 1)  # Blue circles
    
    # Display tracking status
    # status = f"Tracking: {'YES' if best is not None else 'NO'} | Lost: {lost_frames}"
    # cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    

    results = model(frame, verbose=False)
    current_detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0 and conf > 0.4:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # top left, bottom right
                
                box_height = y2 - y1
                if box_height < 30:
                    continue
                
                current_detections.append([x1, y1, x2, y2, conf])

    new_tracked_persons = {}
    used_detections = set()

    for person_id, person_data in tracked_persons.items():
        prev_box = person_data['box']
        best_match_idx = None
        best_iou = 0.3  # Minimum IOU threshold
        
        for idx, detection in enumerate(current_detections):
            if idx in used_detections: 
                # detection already assigned
                # so that same detection not assigned to multiple people
                continue
            
            current_box = detection[:4]
            iou = calculate_iou(prev_box, current_box)
            
            if iou > best_iou:
                best_iou = iou
                best_match_idx = idx
        
        if best_match_idx is not None:
            # Matched existing person
            detection = current_detections[best_match_idx]
            new_tracked_persons[person_id] = {
                'box': detection[:4],
                'conf': detection[4],
                'disappeared': 0
            }
            used_detections.add(best_match_idx)
        else:
            # Person disappeared this frame
            person_data['disappeared'] += 1
            if person_data['disappeared'] < 30:
                new_tracked_persons[person_id] = person_data
            # if exceeds 30 not added to new_tracked_person so deleted
    
    # detection not assigned to any previously tracked person
    for idx, detection in enumerate(current_detections):
        if idx not in used_detections:
            new_tracked_persons[next_person_id] = {
                'box': detection[:4],
                'conf': detection[4],
                'disappeared': 0
            }
            next_person_id += 1
    
    tracked_persons = new_tracked_persons

    redn = 0
    yellown = 0
    for person_id, person_data in tracked_persons.items():
        x1, y1, x2, y2 = person_data['box']
        conf = person_data['conf']
        
        if person_data['disappeared'] == 0:
            if (y2 - y1) > 50:
                team = get_team_color(frame, [x1, y1, x2, y2], red_mask, yellow_mask)
                # xout = x2 < 110 or x1 > 1000
                # yout = y2 > 700 or y2 < 290
                if team == 'red' and x2 > 100 and y2 < 700 and y2 > 390:
                    redn += 1
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                elif team == 'yellow':
                    yellown += 1
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # else:
                #     cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)                    

    
    # Display person count
    count_text = f"RED: {redn} YELLOW: {yellown}"
    cv.putText(frame, count_text, (10, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # Show windows
    cv.imshow('Video', frame)
    # cv.imshow('Mask', mask)
    # cv.imshow('HSV', hsv)
    
    key = cv.waitKey(30) & 0xFF
    if key == ord('d'):
        break
    elif key == ord('p'):  # Pause
        cv.waitKey(0)

capture.release()
cv.destroyAllWindows()