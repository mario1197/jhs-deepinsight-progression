import datetime
import cv2
import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
capture = cv2.VideoCapture(0)

#capture = np.array(capture)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#capture = cv2.flip(capture, 1)
record = False
lst = []
while True:
    ret, frame = capture.read()
    start = time.time()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # print(y)

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Forward"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            #cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(
                    frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                    
    cv2.imshow("Head pose estimation", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 24:
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("save.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        ser = serial.Serial("COM5", 921600, timeout=1)
        
    elif key == 26:
        print("녹화 중지")
        plt.xlim(-90,90)
        plt.ylim(-90,90)
        #plt.plot([0, lst[-1][2]], [0, lst[-1][0]+90],color='skyblue',marker='o', markerfacecolor='blue', markersize=12)
        plt.annotate(text = 'vision', xy=(lst[-1][2], lst[-1][0]+90), xytext=(0, 0), arrowprops=dict(facecolor='red'))
        plt.show()
        plt.close()
        df = pd.DataFrame.from_records(lst)
        df.to_excel('test.xlsx')
        ser.close()
        record = False
        video.release()
        
    if record == True:
        print("녹화 중..")
        video.write(frame)
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('x'):
                print("녹화 중지")
                #plt.xlim(-90,90)
                #plt.ylim(-90,90)
                #plt.plot([0, lst[-1][2]], [0, lst[-1][0]+90],color='skyblue',marker='o', markerfacecolor='blue', markersize=12)
                #plt.annotate(text = 'vision', xy=(lst[-1][2], lst[-1][0]+90), xytext=(0, 0), arrowprops=dict(facecolor='red'))
                #plt.show()
                #plt.close()
                df = pd.DataFrame.from_records(lst)
                df.to_excel('test.xlsx')
                ser.close()
                record = False
                video.release()
                break
            elif ser.in_waiting > 0:
                rx = ser.readline().decode('ascii')   # 아스키 타입으로 읽음
                #i = i+1
                print(rx)
                sample = rx.split(',')
                sample = sample[1:4]
                sample = list(map(float, sample))
                lst.append(sample)
                #cv2.arrowedLine(capture, (320, 240), (320+lst[-1][2], 240+lst[-1][0]+90), (255, 0, 0), 2)
            #key = cv2.waitKey(1)
        

capture.release()
cv2.destroyAllWindows()