#coding:utf-8

import heatmap
import random
import cv2
import numpy as np

def density_heatmap(image, box_centers, radias=100):
    import matplotlib.pyplot as plt
    from colour import Color
    from scipy.spatial import distance
    density_range = 100
    gradient = np.linspace(0, 1, density_range)
    img_width = image.shape[1]
    img_height = image.shape[0]
    density_map = np.zeros((img_height, img_width))
    color_map = np.empty([img_height, img_width, 3], dtype=int)
    # get gradient color using rainbow
    # 使用matplotlib获取颜色梯度
    cmap = plt.get_cmap("rainbow")
    # 使用Color来生成颜色梯度
    blue = Color("blue")
    hex_colors = list(blue.range_to(Color("red"), density_range))
    rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
    for i in range(img_height):
        for j in range(img_width):
            for box in box_centers:
                dist = distance.euclidean(box, (j, i))
                if dist <= radias * 0.25:
                    density_map[i][j] += 10
                elif dist <= radias:
                    density_map[i][j] += (radias - dist) / (radias * 0.75) * 10
            ratio = min(density_range-1, int(density_map[i][j]))
            for k in range(3):
                # color_map[i][j][k] = int(cmap(gradient[ratio])[:3][k]*255)
                color_map[i][j][k] = rgb_colors[ratio][k]
    return color_map

def use_heatmap(image, box_centers):
    import heatmap
    hm = heatmap.Heatmap()
    box_centers = [(i, image.shape[0] - j) for i, j in box_centers]
    #print hm.schemes()
    img = hm.heatmap(box_centers, dotsize=40, size=(image.shape[1], image.shape[0]), opacity=150, scheme = 'classic', area=((0, 0), (image.shape[1], image.shape[0])))
    return img



if __name__ == "__main__":
    pts = []
    firstname = "Camera2_20170614_161550"
    filename_data = firstname + "-data.txt"
    px = []
    py = []
    with open(filename_data, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            px_tmp, py_tmp, pt_temp = [str(i) for i in lines.split()]
            px.append(float(px_tmp))
            py.append(float(py_tmp))
            pts.append((float(px_tmp), float(py_tmp)))
            pass
    pos = np.array(px)
    Efield = np.array(py)
    pts = np.array(pts)
    #print type(pos)
    #print type(px)
    #print pts
    pass

    #for x in range(100):
    #    pts.append((random.random(), random.random()))
    # print "Processing %d points..." % len(pts)
    origin_filename = firstname + ".jpg"
    heatmapback_filename = firstname + "-bg.jpg"
    heatmap_filename = firstname + "-hm.jpg"

    #hm = heatmap.Heatmap()
    #img = hm.heatmap(pts).save("classic1.png")
    print origin_filename
    frame = cv2.imread(origin_filename) # origin image
    heatmap = use_heatmap(frame, pts)
    heatmap.save(heatmapback_filename)
    print heatmapback_filename
    # heatmap2 = density_heatmap(frame, pts, 50)
    # print type(heatmap2)
    # cv2.imwrite("heatmap2",heatmap2)
    # cv2.imshow('heatmap2', heatmap2)
    #heatmap = np.array(heatmap)
    heatmap = cv2.imread(heatmapback_filename) # heatmap image
    overlay = frame.copy()
    alpha = 0.5 # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (200, 0, 0), -1) # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap, alpha, frame, 1-alpha, 0, frame) # 将热度图覆盖到原图
    cv2.imshow('frame', frame)
    cv2.imwrite(heatmap_filename, frame)
    print heatmap_filename
    cv2.waitKey(0)