#encoding=utf8
import os
import argparse
import tensorflow as tf
import cv2
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# def load_model(model_path):
#     config = tf.ConfigProto(allow_soft_placement = True)
#     detection_graph = tf.Graph()
#     with detection_graph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile(model_path, 'rb') as fid:
#             serialized_graph = fid.read()
#             od_graph_def.ParseFromString(serialized_graph)
#             tf.import_graph_def(od_graph_def, name='')
#             sess = tf.Session(graph=detection_graph, config=config)
#     return (detection_graph, sess)

def load_model(model_path):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph

# def object_detection(image_np, detection_graph, sess):
    
#     image_tensor = detection_graph.get_tensor_by_name('input_images:0')
#     #input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images:0')

#     boxes = detection_graph.get_tensor_by_name('model_1/feature_fusion/concat_3:0')
#     #scores = detection_graph.get_tensor_by_name('detection_scores:0')
#     #classes = detection_graph.get_tensor_by_name('detection_classes:0')
#     #num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#     #(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    
#     #boxes = sess.run(boxes, feed_dict={input_images: [image_np]})
#     boxes = sess.run(boxes, feed_dict={image_tensor: [image_np]})
#     return boxes
def sort_poly(p):
    '''
    设四个点的坐标分别为A(x1,y1),B(x2,y2),C(x3,y3),D(x4,y4)
    第一步求xi+yi最小的点pmin。
    第二步按照原来点的顺序从pmin开头。
    第三步A与B之间的垂直距离大于水平距离的话，将顺序调整为ADCB
    '''
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="models/east_icdar2015_resnet_v1_50_rbox_ckpt/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--image", default='/Users/zhangxin/github/EAST/training_samples/img_1.jpg')
    parser.add_argument("--name", default='feature_fusion/concat_3:0')
    parser.add_argument('--out_dir', default='./')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    # (detection_graph, sess) = load_model(model_path)
    detection_graph = load_model(args.frozen_model_filename)
    # We can list operations
    for op in detection_graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')

    im = cv2.imread(args.image)[:, :, ::-1]
    h, w = im.shape[:2]
    new_h, new_w = h / 32 * 32, w / 32 * 32
    im_new = cv2.resize(im, (new_w, new_h))
    print args.image, im.shape, im_new.shape
    # boxes = object_detection(im, detection_graph, sess)
    with tf.Session(graph=detection_graph) as sess:
        # boxes = object_detection(im, detection_graph, sess)
        image_tensor = detection_graph.get_tensor_by_name('input_images:0')
        boxes = detection_graph.get_tensor_by_name(args.name)
        boxes = sess.run(boxes, feed_dict={image_tensor: [im_new]})
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
        print boxes.shape
        for box in boxes:
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            print box.shape
            new_box = [box.astype(np.int32).reshape((-1, 1, 2))]
            cv2.polylines(im_new, new_box, True, color=(255, 0, 0), thickness=1)
            print im_new.shape, new_box
            # break
        img_path = os.path.join(args.out_dir, os.path.basename(args.image))
        cv2.imwrite(img_path, im_new)


