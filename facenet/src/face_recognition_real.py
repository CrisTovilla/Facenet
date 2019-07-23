from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
import sys
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import socket
import traceback
from threading import Thread
from data import DataSerializer
import struct ## new
import facenet
from align import detect_face
from kalman_prediction import Prediction

#input_video=sys.argv[1]
modeldir = 'facenet/src/20180402-114759/'
classifier_filename = 'facenet/src/20180402-114759/my_classifier.pkl'
npy=''
train_img="facenet/dataset/raw"

"""NN vars"""
pnet = ""
rnet = "" 
onet = "" 
sess = ""
embedding_size = ""
phase_train_placeholder = ""
images_placeholder  = ""
embeddings = ""
model = ""
HumanNames  = ""
"""|||||||"""

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def start_server():
    host = socket.gethostbyname(socket.gethostname()) 
    #host = "192.168.3.245"
    port = 8888         # arbitrary non-privileged port

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state, without waiting for its natural timeout to expire
    print("Socket created")

    try:
        soc.bind((host, port))
    except:
        print("Bind failed. Error : " + str(sys.exc_info()))
        sys.exit()

    soc.listen(5)       # queue up to 5 requests
    print("Started server at: ", host, ":", port)
    print("Socket now listening")

    # infinite loop- do not reset for every requests
    while True:
        connection, address = soc.accept()
        ip, port = str(address[0]), str(address[1])
        print("Connected with " + ip + ":" + port)

        try:
            Thread(target=client_thread, args=(connection, ip, port)).start()
        except:
            print("Thread did not start.")
            traceback.print_exc()

    soc.close()


def client_thread(connection, ip, port, max_buffer_size = 4096):    
    is_active = True
    kalman_predictions = []
    kalman_keys = []

 
    while is_active:
        client_input = receive_input(connection, max_buffer_size)
        detected_name, detected_pos, detected_precision, kalman_predictions, kalman_keys = detect(client_input, kalman_predictions, kalman_keys)

        if "--QUIT--" in client_input:
            print("Client is requesting to quit")
            connection.close()
            print("Connection " + ip + ":" + port + " closed")
            is_active = False
        else:
            image= client_input

            result, frame = cv2.imencode('.jpg', image, encode_param)
            ''' Creating a SerializerData that includes a message and frame '''
            send_data = DataSerializer("Saludos del server", detected_name, detected_pos, detected_precision)
            data = pickle.dumps(send_data, 0)
            size = len(data)

            connection.sendall(struct.pack(">L", size) + data)


def receive_input(connection, max_buffer_size):
    data = b""
    payload_size = struct.calcsize(">L")
    temp = len(data)
    disconnected_socket = False
    while len(data) < payload_size:
        print("Primer while, Recv: {}".format(len(data)))
        data += connection.recv(4096)
        if temp == len(data):
            disconnected_socket = True
            break
        temp = len(data)

    if not disconnected_socket:        
        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            print("Segundo while")
            data += connection.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        recv_data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = recv_data.frame
        print(recv_data.msg)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        result = process_input(frame)

        return frame 
    else:
        return ['--QUIT--']


def process_input(input_str):
    print("Processing the input received from client")

    return "Hello " + str(input_str).upper()


def resize_frame(frame, width, height):
    if width > 1700:
        #1920 / 720 = 2.66
        #2.66 * 100 = 260
        #138 - 100= 38
        #100 - 38 = 62
        percent = (width / 1700) * 100
        width = int(frame.shape[1] * 100 / percent)
        height = int(frame.shape[0] * 100 / percent)
        dim = (width, height)        
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    return frame

def detect(frame, kalman_predictions, kalman_keys):
    c = 0
    minsize = 110  # minimum size of face
    threshold = [0.8, 0.8, 0.8]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 32
    frame_interval = 10
    image_size = 160
    input_image_size = 160
    detected_name = []
    detected_pos = []
    detected_precision = []

    #video_capture = cv2.VideoCapture(1)

    #frame = video(video_capture)

    if (c % frame_interval == 0):

        """Optmizacion"""
        # if frame.ndim == 2:                
        #     frame = facenet.to_rgb(frame)

        """Duda"""
        #frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Detected_FaceNum: %d' % nrof_faces)   
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            #img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces,4), dtype=np.int32)                    
            for i in range(nrof_faces):
                continuar = True
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]                
                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('Face is very close!')
                    continue
                                        
                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                '''Checar aqui para que salte el face is very close'''
                if len(cropped) <= i:
                    print(len(cropped), " | ",i)
                    continue
                
                cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                #print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                # print("predictions")
                #print(best_class_indices,' with accuracy ',best_class_probabilities)
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)    #boxing face                        
                # print(best_class_probabilities)
                detected_name.append(HumanNames[best_class_indices[0]])
                detected_pos.append(bb[i])
                detected_precision.append(best_class_probabilities)
                if best_class_probabilities>0.88:                    
                    #print("Persona detectada: ", HumanNames[best_class_indices[0]])
                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    #print(HumanNames)
                    for H_i in HumanNames:
                        if HumanNames[best_class_indices[0]] == H_i:
                            result_names = HumanNames[best_class_indices[0]]
                            if result_names in kalman_keys:
                                print("Ese nombre ya existe, index: ", kalman_keys.index(result_names))
                            else:
                                kalman_keys.append(result_names)
                                print("Agregando nuevo usuario: ", result_names)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        2, (0, 0, 255), thickness=2, lineType=2)
                elif best_class_probabilities>0.70:
                    print("Persona detectada: ", HumanNames[best_class_indices[0]])
                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    #print(HumanNames)
                    for H_i in HumanNames:
                        if HumanNames[best_class_indices[0]] == H_i:
                            result_names = HumanNames[best_class_indices[0]]
                            mensaje = str("Duda: " + result_names)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 255), 2)    #boxing face
                            cv2.putText(frame, mensaje, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1.5, (0, 0, 255), thickness=2, lineType=2)
        else:
            print('Alignment Failure')
        print("\n")
    # c+=1
    #out.write(frame)
    print(detected_name)
    print(detected_pos)
    print(detected_precision)
    return detected_name, detected_pos, detected_precision, kalman_predictions, kalman_keys
    #resized_frame = resize_frame(frame, frame.shape[1], frame.shape[0])
    #cv2.imshow('Video', resized_frame)
    #cv2.waitKey(1)

    #video_capture.release()
    #out.release()

def video(video_capture):    
    #width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    #height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    ret, frame = video_capture.read()

    return frame
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
    #out = cv2.VideoWriter('output/'+input_video.split('/')[-1],fourcc, 25.0, (width,height))    

def initialize():
    global pnet
    global rnet 
    global onet 
    global sess 
    global embedding_size
    global phase_train_placeholder
    global images_placeholder 
    global embeddings
    global model
    global HumanNames 

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():            
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)            

            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model('facenet/src/20180402-114759/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)                    

            print('Start Recognition')
            prevTime = 0                        

    cv2.destroyAllWindows()
    
def main():
    initialize()
    start_server()
    

if __name__ == '__main__':
    main()