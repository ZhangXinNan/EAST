import tensorflow as tf
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0't

def load_model(model_path):
    config = tf.ConfigProto(allow_soft_placement = True)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph, config=config)
    return (detection_graph, sess)


def object_detection(image_np, detection_graph, sess):
    
    image_tensor = detection_graph.get_tensor_by_name('input_images:0')
    #input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images:0')

    boxes = detection_graph.get_tensor_by_name('model_1/feature_fusion/concat_3:0')
    #scores = detection_graph.get_tensor_by_name('detection_scores:0')
    #classes = detection_graph.get_tensor_by_name('detection_classes:0')
    #num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    
    # boxes = sess.run(boxes, feed_dict={input_images: [image_np]})
    boxes = sess.run(boxes, feed_dict={image_tensor: [image_np]})
    return boxes



if __name__ == '__main__':
    model_path = 'models/east_icdar2015_resnet_v1_50_rbox/freeze_model01.pb'
    im_fn = '/Users/zhangxin/github/EAST/demo/010.png'
    (detection_graph, sess) = load_model(model_path)

    input_images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_images')

    im = cv2.imread(im_fn)[:, :, ::-1]
    print im_fn, im.shape
    # h, w = im.shape[:2]
    # new_h, new_w = h / 32 * 32, w / 32 * 32
    # im = cv2.resize(im, (new_w, new_h))
    # print im_fn, im.shape
    boxes = object_detection(im, detection_graph, sess)
    print boxes
