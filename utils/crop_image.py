import numpy as np
from PIL import Image
from scipy.misc import imresize


def crop_image_blur(img, bbox):
    x, y, w, h = np.array(bbox, dtype='float32')
    img_h, img_w, _ = img.shape
    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        cropped = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    return cropped

def move_crop(pos_, deta_pos, img_size, rate):
    flag = 0
    if pos_.shape.__len__() == 1:
        pos_ = np.array(pos_).reshape([1, 4])
        deta_pos = np.array(deta_pos).reshape([1, 3])
        flag = 1
    pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]#
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] * (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])


    pos[pos[:, 2] < 10, 2] = 10
    pos[pos[:, 3] < 10, 3] = 10

    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] + pos[:, 2] > img_size[0], 0] = \
        img_size[0] - pos[pos[:, 0] + pos[:, 2] > img_size[0], 2] - 1
    pos[pos[:, 1] + pos[:, 3] > img_size[1], 1] = \
        img_size[1] - pos[pos[:, 1] + pos[:, 3] > img_size[1], 3] - 1
    pos[pos[:, 0] < 0, 0] = 0
    pos[pos[:, 1] < 0, 1] = 0

    pos[pos[:, 0] > img_size[1], 0] = img_size[1]
    pos[pos[:, 1] > img_size[0], 1] = img_size[0]
    if flag == 1:
        pos = pos[0]

    return pos

def move_crop_tracking(pos_, deta_pos, img_size, rate):
    flag = 0
    if pos_.shape.__len__() == 1:
        pos_ = np.array(pos_).reshape([1, 4])
        deta_pos = np.array(deta_pos).reshape([1, 3])
        flag = 1
    pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]#
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] * (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])

    if np.max((pos[:, 3] > (pos[:, 2] / rate) * 1.2)) == 1.0:
        pos[:, 3] = pos[:, 2] / rate

    if np.max((pos[:, 3] < (pos[:, 2] / rate) / 1.2)) == 1.0:
        pos[:, 2] = pos[:, 3] * rate

    pos[pos[:, 2] < 10, 2] = 10
    pos[pos[:, 3] < 10, 3] = 10

    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] > img_size[1], 0] = img_size[1]
    pos[pos[:, 1] > img_size[0], 1] = img_size[0]
    pos[pos[:, 0] < -pos[:, 2], 0] = -pos[:, 2]
    pos[pos[:, 1] < -pos[:, 3], 1] = -pos[:, 2]

    if flag == 1:
        pos = pos[0]

    return pos


def crop_image(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h
    out_flag = 0
    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        try:
            cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
        except:
        #     a= 1
            out_flag = 1

    scaled_l = imresize(cropped, (img_size, img_size))



    return scaled_l

def crop_image_actor(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h
    out_flag = 0
    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_l = imresize(cropped, (img_size, img_size))

    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)


    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        if max(abs(min_y - min_y_val) / half_h, abs(max_y - max_y_val) / half_h, abs(min_x - min_x_val) / half_w, abs(max_x - max_x_val) / half_w) > 0.3:
            out_flag = 1
        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_g = imresize(cropped, (img_size, img_size))

    return scaled_l, scaled_g, out_flag

def crop_image_actor_(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h
    out_flag = 0
    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]
        # print("+++++++++++111OK+++++++++++++++++++++")
    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        if (min_y_val - min_y) < (max_y_val - min_y):

            cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
            #
            # print("11111111+++++++++++++++")
        else:
            # pass
            out_flag = 1
            # print("22222222---------------")

    scaled_l = np.array(Image.fromarray(cropped).resize((img_size, img_size)))

    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]
        # print("+++++++++++222OK+++++++++++++++++++++")
    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        if (min_y_val - min_y) < (max_y_val - min_y):

            cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

            # print("+++++++++++++++")
        else:
            # pass
            out_flag = 1
            # print("---------------")

    scaled_g = np.array(Image.fromarray(cropped).resize((img_size, img_size)))

    return scaled_l, scaled_g, out_flag
