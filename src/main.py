import cv2
import numpy as np
import tensorflow as tf
import time
import collections
import six
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import _get_multiplier_for_color_randomness
from imutils.video import WebcamVideoStream
import imutils

STANDARD_COLORS = [
  'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
  'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
  'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
  'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
  'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
  'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
  'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
  'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
  'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
  'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
  'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
  'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
  'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
  'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
  'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
  'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
  'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
  'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
  'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
  'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
  'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
  'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
  'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def fire_detect(frame, model_alex, model_cascade):
    fire = model_cascade.detectMultiScale2(frame, 1.1, 5)
    result = []
    for [x,y,w,h] in fire[0]:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color,(100,100))
        data = []
        image = np.array(roi_color)
        data.append(image)
        data = np.array(data, dtype="float32") / 255.0

        ypred = model_alex.predict(data)
        if np.argmax(ypred) == 0:
            #cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            #return (x-20, y-20, x+w+20, y+h+20)
            result.append([[int(ypred[0][0]*100)], [x, y, x+w, y+h]])
    return result
    

def run_inference_for_single_image(model, image):
  
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def visualize_boxes_and_labels_on_image_array(
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    track_ids=None,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_keypoint_scores_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if keypoint_scores is not None:
                box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(round(100*scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = 'ID {}'.format(track_ids[i])
                    else:
                        display_str = '{}: ID {}'.format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
    box_class = []
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        box_class.append([ymin, xmin, ymax, xmax])

    return box_class

def detect_frame(frame, model, category_index):
    image_np = np.array(frame)
    output_dict = run_inference_for_single_image(model, image_np)
    boxes = visualize_boxes_and_labels_on_image_array(
            output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'],
            category_index, instance_masks=output_dict.get('detection_masks_reframed', None))
    return boxes #ymin xmin ymax xmax

def normal(frame):
    cv2.rectangle(frame, (0, 0), (640, 20), (0,255,0), -1)
    cv2.putText(frame, "Normal", (280, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def fire(frame, locate_of_flame, accuracy):
    cv2.rectangle(frame, (0, 0), (640, 20), (0,0,255), -1)
    cv2.putText(frame, "FIRE", (290, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame,(locate_of_flame[0], locate_of_flame[1]),(locate_of_flame[2],locate_of_flame[3]),(0,0,255),2)
    cv2.putText(frame, "FIRE " + str(accuracy) + "%", (locate_of_flame[0], locate_of_flame[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    return frame

def fake_fire(frame, locate_of_flame, accuracy):
    cv2.rectangle(frame, (0, 0), (640, 20), (0,255,0), -1)
    cv2.putText(frame, "FAKE FIRE", (290, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame,(locate_of_flame[0], locate_of_flame[1]),(locate_of_flame[2],locate_of_flame[3]),(0,255,0),2)
    cv2.putText(frame, "FAKE FIRE " + str(accuracy) + "%", (locate_of_flame[0], locate_of_flame[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def checkIntersection(boxA, boxB):
    
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    return(foundIntersect, [x, y, w, h])

def verify_same_box(a, b, num_check):

    if (((a[0] - b[0]) <= num_check) and ((a[1] - b[1]) <= num_check) and ((a[2] - b[2]) <= num_check) and ((a[3] - b[3]) <= num_check)):
        return True
    else:
        return False

def main():
    fire_cascade = cv2.CascadeClassifier('model/fire_detection.xml')
    alexnet = tf.keras.models.load_model("model/fire-detect.h5")
    model = tf.saved_model.load("model/ssd_mobilenet_v2/saved_model")
    category_index = label_map_util.create_category_index_from_labelmap("model/ssd_mobilenet_v2/label_map.txt", use_display_name=True)

    # live_Camera = cv2.VideoCapture(0)
    # live_Camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # live_Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # live_Camera.set(cv2.CAP_PROP_FPS, 30)
    new_frame_time = 0
    prev_frame_time = 0
    overlap = [0,0,0,0]
    check_flame_in_frame = 0
    vs = WebcamVideoStream(src=0).start()

    out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10, (640, 480))
    while(True):
        # ret, frame = live_Camera.read()
        frame = vs.read()
	    # frame = imutils.resize(frame, width=640, height=480)
        frame = cv2.resize(frame, (640, 480))
        locate_of_list_flame = fire_detect(frame, alexnet, fire_cascade)
        locate_of_list_frame = detect_frame(frame, model, category_index)

        if len(locate_of_list_flame) == 0:
            frame = normal(frame)
            check_flame_in_frame = 0
        else:
            for locate_of_flame in locate_of_list_flame:
                if len(locate_of_list_frame) == 0:
                    frame = fire(frame, locate_of_flame[1], locate_of_flame[0])
                else:
                    check = 0
                    foundIntersectList = []
                    for object in locate_of_list_frame:
                        locate = [int(object[1]*640), int(object[0]*480), int(object[3]*640), int(object[2]*480)]
                        foundIntersect, locate = checkIntersection(locate_of_flame[1], locate)
                        foundIntersectList.append(foundIntersect)

                        if True in foundIntersectList:
                            if verify_same_box(locate, locate_of_flame[1], np.abs(1)):
                                if check_flame_in_frame == 5:
                                    overlap = locate_of_flame[1]
                                    if verify_same_box(overlap, locate_of_flame[1], np.abs(5)):
                                        frame = fake_fire(frame, locate_of_flame[1], locate_of_flame[0])
                                        check = 0
                                    else:
                                        frame = fire(frame, locate_of_flame[1], locate_of_flame[0])
                                        check = 1
                                    check_flame_in_frame = 0
                                else:
                                    if check == 0:
                                        frame = fake_fire(frame, locate_of_flame[1], locate_of_flame[0])
                                    if check == 1:
                                        frame = fire(frame, locate_of_flame[1], locate_of_flame[0])
                            else:  
                                new_flame = [0,0,0,0]
                                if check_flame_in_frame == 5:
                                    new_flame = list(locate_of_flame[1])
                                    if verify_same_box(new_flame, list(locate_of_flame[1]), np.abs(5)):
                                        frame = fake_fire(frame, locate_of_flame[1], locate_of_flame[0])
                                        check = 0
                                    else:
                                        frame = fire(frame, locate_of_flame[1], locate_of_flame[0])
                                    check_flame_in_frame = 0
                                else:
                                    if check == 0:
                                        frame = fake_fire(frame, locate_of_flame[1], locate_of_flame[0])
                                    if check == 1:
                                        frame = fire(frame, locate_of_flame[1], locate_of_flame[0])
                        else:
                            frame = fire(frame, locate_of_flame[1], locate_of_flame[0])  
                                        
                                    
        check_flame_in_frame += 1 
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = str(int(fps)) + "FPS"
        cv2.putText(frame, fps, (7, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            vs.stop()
            break

if __name__ == "__main__":
    main()
