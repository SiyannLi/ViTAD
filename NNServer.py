import socket
import struct
import threading
import numpy as np
import cv2
from drive import AutoDrive

# 定义常量
DEFAULT_PORT = 12345
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
IMAGE_CHANNELS = 3
NUM_IMAGES = 3
DEFAULT_BUFLEN = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES + 1024

# FVector 结构
class FVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.X = x
        self.Y = y
        self.Z = z

# DataPackage 结构
class DataPackage:
    def __init__(self):
        self.Steering = 0.0
        self.Throttle = 0.0
        self.Brake = 0.0
        self.WheelAngle = 0.0
        self.Timestamp = 0.0
        self.Acceleration = 0.0
        self.Direction = 0
        self.Position = FVector()
        self.Velocity = 0.0
        self.Heading = 0.0
        self.ImageData = bytearray()

# 显示图片
def show_image(data):
    combined_height = IMAGE_HEIGHT * 3
    combined_width = IMAGE_WIDTH * 3
    combined_image = np.zeros((combined_height, combined_width, IMAGE_CHANNELS), dtype=np.uint8)

    positions = [
        (0, 0), (IMAGE_WIDTH, 0), (2 * IMAGE_WIDTH, 0),  # 第一行
        (IMAGE_WIDTH // 2, IMAGE_HEIGHT), (IMAGE_WIDTH + IMAGE_WIDTH // 2, IMAGE_HEIGHT)  # 第二行
    ]

    for i in range(NUM_IMAGES):
        image_data = np.frombuffer(data[i * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS:(i + 1) * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], dtype=np.uint8)
        image = image_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        x, y = positions[i]
        combined_image[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH] = image

    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(1)

# 处理数据
def process_data(data_package):
    show_image(data_package.ImageData)
    print(data_package.Direction, data_package.Position.X, data_package.Position.Y, data_package.Position.Z)

# 反序列化数据包
def deserialize_vehicle_data(data):
    package = DataPackage()
    offset = 0

    total_image_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES
    package.ImageData = data[:total_image_size]
    offset += total_image_size

    package.Position = FVector(*struct.unpack('ddd', data[offset:offset + 24]))
    offset += 24

    package.Velocity, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Heading, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Direction, = struct.unpack('i', data[offset:offset + 4])
    offset += 4

    package.Acceleration, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Timestamp, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.WheelAngle, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Brake, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Throttle, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Steering, = struct.unpack('f', data[offset:offset + 4])

    return package

# 处理客户端连接
def handle_client(client_socket, auto_drive):
    recvbuf = bytearray(DEFAULT_BUFLEN)

    while True:
        try:
            # 接收数据包
            bytes_received = client_socket.recv_into(recvbuf)
            if bytes_received > 0:
                data_package = deserialize_vehicle_data(recvbuf)

                # 检查数据包完整性
                total_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES
                if len(data_package.ImageData) != total_size:
                    print(f"Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes")
                    response = f"Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes"
                    client_socket.sendall(response.encode('utf-8'))
                else:
                    # 使用 AutoDrive 处理图像，生成控制信号
                    steering, throttle, brake = process_client_data(data_package, auto_drive)
                    response = f"0.3;0;{3*steering}"
                    print(f"[INFO] Prediction: Throttle={0.7},  Brake={0.0},Steering={steering}")

                    # response = "1;0;0"
                    client_socket.sendall(response.encode('utf-8'))
            else:
                print("Connection closing...")
                break
        except Exception as e:
            print(f"Exception: {e}")
            break

    client_socket.close()

# 从数据包中处理图像并调用 AutoDrive
def process_client_data(data_package, auto_drive):
    IMAGE_W = IMAGE_WIDTH
    IMAGE_H = IMAGE_HEIGHT
    IMAGE_C = IMAGE_CHANNELS

    try:
        # 获取第二张图片（中间摄像头图像）
        second_image_start = IMAGE_W * IMAGE_H * IMAGE_C
        second_image_end = 2 * second_image_start
        second_image_data = data_package.ImageData[second_image_start:second_image_end]

        # 解码为 NumPy 数组
        second_image = np.frombuffer(second_image_data, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W, IMAGE_C)

        # 使用 AutoDrive 模块预测控制信号
        return auto_drive.predict_control(second_image)
    except Exception as e:
        print(f"[ERROR] Failed to process client data: {e}")
        return 0.0, 0.0, 1.0  # 默认值

# 主函数
def main():
    # 初始化 AutoDrive 模块
    
    model_path = 'D:\\Studium_in_DE\\TUM\\Dritte_Semester\\Repository\\End2EndAD\\wuyang_2\\trained_model2_1_0.1091.pth'  # 模型文件路径
    auto_drive = AutoDrive(model_path)

    # 启动服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', DEFAULT_PORT))
    server_socket.listen()

    print(f"Waiting for client connection on port {DEFAULT_PORT}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Client connected from {addr}")

        # 创建客户端处理线程
        client_handler = threading.Thread(target=handle_client, args=(client_socket, auto_drive))
        client_handler.start()

if __name__ == "__main__":
    main()