import cv2
import numpy as np


def handle_pose(output, input_shape):
    heatmaps = output['Mconv7_stage2_L2']
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):
    text_classes = output['model/segm_logits/add']
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    return out_text


def handle_car(output, input_shape):
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    color_pred = np.argmax(color)
    type_pred = np.argmax(car_type)
    
    return color_pred, type_pred


def handle_output(model_type):
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


def preprocessing(Type, input_image, height, width):
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
