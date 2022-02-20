from tqdm import tqdm
import numpy as np
from imutils import face_utils
# import dlib
from collections import OrderedDict
import cv2
from calib_utils import track_bidirectional
import mediapipe as mp

def shape_to_face(shape, face, width, height, scale):
    x_min, x_max = face[0], face[3]
    y_min, y_max = face[1], face[4]
    z_min, z_max = face[2], face[5]
    
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    face_size = int(max(x_max - x_min, y_max - y_min) * scale)
    face_size = face_size // 2 * 2
    
    x1 = max(x_center - face_size // 2, 0)
    y1 = max(y_center - face_size // 2, 0)
    
    face_size = min(width - x1, face_size)
    face_size = min(height - y1, face_size)
    
    x2 = x1 + face_size
    y2 = y1 + face_size
    
    face_new = [int(x1) , int(y1) , z_min, int(x2), int(y2), z_max]
    
    return face_new, face_size

def predict_single_frame(frame, frame_width, frame_height):
  # use mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
    
    shape = []
    face = []
    x_min, x_max = frame_width, 0
    y_min, y_max = frame_height, 0
    z_min, z_max = 1.0, 0.0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id,lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
                z_min, z_max = min(z_min, lm.z), max(z_max, lm.z)
                shape.append([x, y, lm.z])
    
        shape = np.array(shape)
        face = [x_min, y_min, z_min, x_max, y_max, z_max]
    
    # print(shape)
    # print(type(shape)) # numpy.ndarray
    # print(shape.shape) # (468, 3)
    # print(len(face)) # 6
    return face, shape

def check_and_merge(location, forward, feedback, P_predict, status_fw=None, status_fb=None):
    num_pts = 468
    check = [True] * num_pts
    
    target = location[1]
    forward_predict = forward[1]
    
    # To ensure the robustness through feedback-check
    forward_base = forward[0]  # Also equal to location[0]
    feedback_predict = feedback[0]
    feedback_diff = feedback_predict - forward_base
    feedback_dist = np.linalg.norm(feedback_diff, axis=1, keepdims=True)
    
    # For Kalman Filtering
    detect_diff = location[1] - location[0]
    detect_dist = np.linalg.norm(detect_diff, axis=1, keepdims=True)
    predict_diff = forward[1] - forward[0]
    predict_dist = np.linalg.norm(predict_diff, axis=1, keepdims=True)
    predict_dist[np.where(predict_dist == 0)] = 1  # Avoid nan
    P_detect = (detect_dist / predict_dist).reshape(num_pts)
    
    for ipt in range(num_pts):
        if feedback_dist[ipt] > 2:  # When use float
            check[ipt] = False
        
    if status_fw is not None and np.sum(status_fw) != num_pts:
        for ipt in range(num_pts):
            if status_fw[ipt][0] == 0:
                check[ipt] = False
    if status_fw is not None and np.sum(status_fb) != num_pts:
        for ipt in range(num_pts):
            if status_fb[ipt][0] == 0:
                check[ipt] = False
    location_merge = target.copy()
    # Merge the results:
    """
    Use Kalman Filter to combine the calculate result and detect result.
    """
    
    Q = 0.3  # Process variance
    
    for ipt in range(num_pts):
        if check[ipt]:
            # Kalman parameter
            P_predict[ipt] += Q
            K = P_predict[ipt] / (P_predict[ipt] + P_detect[ipt])
            location_merge[ipt] = forward_predict[ipt] + K * (target[ipt] - forward_predict[ipt])
            # Update the P_predict by the current K
            P_predict[ipt] = (1 - K) * P_predict[ipt]
    return location_merge, check, P_predict

def detect_frames_track(frames, fps, use_visualization, visualize_path, video):

    frames_num = len(frames)
    frame_height, frame_width = frames[0].shape[:2]
    
    face_size_normalized = 400
    faces = []
    locations_1 = []
    locations_2 = []
    locations_3 = []
    shapes_origin = []
    shapes_para = []  # Use to recover the shape in whole frame. ([x1, y1, scale_shape])
    face_size = 0
    skipped = 0

    """
    Use single frame to detect face on Mediapipe (CPU)
    """
    # ----------------------------------------------------------------------------#

    print("Detecting:")
    
    for i in tqdm(range(frames_num)):
        frame = frames[i]
        face, shape = predict_single_frame(frame, frame_width, frame_height)
        
        if face:
            face_new, face_size = shape_to_face(shape, face, frame_width, frame_height, 1.2) # face_new: original size
            # print(face_new)
        
            faceFrame = frame[face_new[1]: face_new[4], # y_min : y_max
                          face_new[0]: face_new[3]] # x_min : x_max
            
            if face_size < face_size_normalized:
                inter_para = cv2.INTER_CUBIC
            else:
                inter_para = cv2.INTER_AREA
            
            face_norm = cv2.resize(faceFrame, (face_size_normalized, face_size_normalized), interpolation=inter_para)
            scale_shape = face_size_normalized/face_size
            
            shape_norm = (shape-np.array([face[0], face[1], 0])) * scale_shape
            faces.append(face_norm)
            shapes_para.append([face_new[0], face_new[1], scale_shape])
            shapes_origin.append(shape)
            
            # print(type(shape_norm)) # numpy.ndarray
            # print(shape_norm.shape) # (468,3)
            shape_norm_1 = shape_norm[ : ,2:3]
            shape_norm_2 = shape_norm[ : , :2]
            
            locations_1.append(shape_norm_1)
            locations_2.append(shape_norm_2)
            locations_3.append(shape_norm) # list
            
            

    """
    Calibration module.
    """
    segment_length = 2
    locations_sum = len(locations_2)
    if locations_sum == 0:
            return []
    locations_track = [locations_2[0]]
    num_pts = 468
    P_predict = np.array([0] * num_pts).reshape(num_pts).astype(float)
    
    print("Tracking")
    # for i in range(locations_sum - 1):
    for i in tqdm(range(locations_sum - 1)):
        faces_seg = faces[i:i + segment_length]
        locations_seg = locations_2[i:i + segment_length]
        """
        OpenCV Version
        """
        
        lk_params = dict(winSize=(15, 15),
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Use the tracked current location as input. Also use the next frame's predicted location for
        # auxiliary initialization.
        
        # print(type(locations_track)) # list
        # print(type(locations_track[i])) # numpy.ndarray
        
        start_pt = locations_track[i].astype(np.float32)
        target_pt = locations_seg[1].astype(np.float32)
        
        forward_pt, status_fw, err_fw = cv2.calcOpticalFlowPyrLK(faces_seg[0], faces_seg[1],
                                                                start_pt, target_pt, **lk_params,
                                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        feedback_pt, status_fb, err_fb = cv2.calcOpticalFlowPyrLK(faces_seg[1], faces_seg[0],
                                                                forward_pt, start_pt, **lk_params,
                                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        
        forward_pts = [locations_track[i].copy(), forward_pt]
        feedback_pts = [feedback_pt, forward_pt.copy()]
        
        forward_pts = np.rint(forward_pts).astype(int)
        feedback_pts = np.rint(feedback_pts).astype(int)
        
        merge_pt, check, P_predict = check_and_merge(locations_seg, forward_pts, feedback_pts, P_predict, status_fw, status_fb)
        
        locations_track.append(merge_pt)
    
    aligned_landmarks = []
    
    for i in range(len(locations_track)):
        # print(locations_track[i].shape) # (468,2)
        # print(locations_1[i].shape) # (468,1)
        shape_new = np.concatenate((locations_track[i], locations_1[i]), axis=1)
        # print(shape_new.shape) # (468,3)
        
        shape_new = shape_new.ravel()
        shape_new = shape_new.tolist()
        aligned_landmarks.append(shape_new)
    
    return aligned_landmarks