import cv2
import numpy as np
import time
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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

def camera_detect(model, category_index):
    live_Camera = cv2.VideoCapture(0)
    live_Camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    live_Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    live_Camera.set(cv2.CAP_PROP_FPS, 30)
    new_frame_time = 0
    prev_frame_time = 0
    while(live_Camera.isOpened()):
        ret, frame = live_Camera.read()
        image_np = np.array(frame)
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=1)
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = str(int(fps)) + "FPS"
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 2)
        cv2.imshow("Fire Detection",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    live_Camera.release()

    cv2.destroyAllWindows()

def main():
    model = tf.saved_model.load("model/ssd_mobilenet_v2/saved_model")
    category_index = label_map_util.create_category_index_from_labelmap("model/ssd_mobilenet_v2/label_map.txt", use_display_name=True)
    camera_detect(model, category_index)

if __name__ == "__main__":
    main()