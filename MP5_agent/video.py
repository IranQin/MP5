import cv2
import os
import re

def images_to_video(image_folder, video_name, fps):
    # 获取图片列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # 提取文件名中的数字并转换为整数，然后按照这个数字排序
    images.sort(key=lambda img: int(re.findall(r'\d+', img)[0]))

    # 读取第一张图片以获取视频的维度
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码，FPS 和大小
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


images_to_video("mp5/video copy 4", "video copy 4.mp4", 15)