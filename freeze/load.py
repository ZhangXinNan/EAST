#encoding=utf8
import os
import argparse
import tensorflow as tf
import cv2

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="models/east_icdar2015_resnet_v1_50_rbox/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--image", default='/Users/zhangxin/github/EAST/demo/010.png')
    parser.add_argument("--name", default='feature_fusion/concat_3:0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()    

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
    print args.image, im.shape
    h, w = im.shape[:2]
    new_h, new_w = h / 32 * 32, w / 32 * 32
    im = cv2.resize(im, (new_w, new_h))
    print args.image, im.shape
    # boxes = object_detection(im, detection_graph, sess)
    with tf.Session(graph=detection_graph) as sess:
        # boxes = object_detection(im, detection_graph, sess)
        image_tensor = detection_graph.get_tensor_by_name('input_images:0')
        boxes = detection_graph.get_tensor_by_name(args.name)
        boxes = sess.run(boxes, feed_dict={image_tensor: [im]})
        print boxes

