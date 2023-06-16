from flask import Flask, render_template, Response
import socket
import cv2
import pickle
import struct

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Assuming you have an HTML template to display the video stream

@app.route('/video')
def video():
    def generate_frames():
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_ip = '172.18.12.84'  # Paste your server IP address here
        port = 9999
        client_socket.connect((host_ip, port))

        data = b""
        payload_size = struct.calcsize("Q")

        while True:
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024)  # 4K
                if not packet:
                    break
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4*1024)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)

            # Convert the frame to a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        client_socket.close()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
