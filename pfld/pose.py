import numpy as np
import cv2
import torch

def calculate_pitch_yaw_roll(landmarks_2D, cam_w=384, cam_h=384, radians=False):
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    LEFT_EYEBROW_LEFT = [6.825897, 6.760612, 4.402142]
    LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_RIGHT = [-6.825897, 6.760612, 4.402142]
    LEFT_EYE_LEFT = [5.311432, 5.485328, 3.987654]
    LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    RIGHT_EYE_RIGHT = [-5.311432, 5.485328, 3.987654]
    NOSE_LEFT = [2.005628, 1.409845, 6.165652]
    NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    MOUTH_RIGHT = [-2.774015, -2.080775, 5.048531]
    LOWER_LIP = [0.000000, -3.116408, 6.097667]
    CHIN = [0.000000, -7.415691, 4.070434]
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    landmarks_3D = np.float32([LEFT_EYEBROW_LEFT,
                                    LEFT_EYEBROW_RIGHT,
                                    RIGHT_EYEBROW_LEFT,
                                    RIGHT_EYEBROW_RIGHT,
                                    LEFT_EYE_LEFT,
                                    LEFT_EYE_RIGHT,
                                    RIGHT_EYEBROW_LEFT,
                                    RIGHT_EYEBROW_RIGHT,
                                    NOSE_LEFT,
                                    NOSE_RIGHT,
                                    MOUTH_LEFT,
                                    MOUTH_RIGHT,
                                    LOWER_LIP,
                                    CHIN])
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat,tvec))
    _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = map(lambda temp: temp[0], euler_angles)
    return pitch, yaw, roll



def get_euler_angle_weights(landmarks_batch, euler_angles_pre, device):
    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

    euler_angles_landmarks = []
    landmarks_batch = landmarks_batch.cpu().numpy()
    for index in TRACKED_POINTS:
        euler_angles_landmarks.append(landmarks_batch[:, 2 * index:2 * index + 2])
    euler_angles_landmarks = np.asarray(euler_angles_landmarks).transpose((1, 0, 2)).reshape((-1, 28))

    euler_angles_gt = []
    for j in range(euler_angles_landmarks.shape[0]):
        pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmarks[j])
        euler_angles_gt.append((pitch, yaw, roll))
    euler_angles_gt = np.asarray(euler_angles_gt).reshape((-1, 3))

    euler_angles_gt = torch.Tensor(euler_angles_gt).to(device)
    euler_angle_weights = 1 - torch.cos(torch.abs(euler_angles_gt - euler_angles_pre))
    euler_angle_weights = torch.sum(euler_angle_weights, 1)

    return euler_angle_weights