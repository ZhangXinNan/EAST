```
➜  EAST git:(zxdev_mac_service) ✗ python test_single.py 
2018-04-02 21:11:04.129392: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
/Users/zhangxin/github/EAST/demo/010.png (250, 300, 3)
Traceback (most recent call last):
  File "test_single.py", line 49, in <module>
    boxes = object_detection(im, detection_graph, sess)
  File "test_single.py", line 31, in object_detection
    boxes = sess.run(boxes, feed_dict={image_tensor: [image_np]})
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1128, in _run
    feed_dict_tensor, options, run_metadata)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1344, in _do_run
    options, run_metadata)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1363, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Number of ways to split should evenly divide the split dimension, but got split_dim 0 (size = 1) and num_split 2
	 [[Node: prefix/split = Split[T=DT_FLOAT, num_split=2, _device="/job:localhost/replica:0/task:0/device:CPU:0"](prefix/split/split_dim, _arg_prefix/input_images_0_0)]]

Caused by op u'prefix/split', defined at:
  File "test_single.py", line 39, in <module>
    (detection_graph, sess) = load_model(model_path)
  File "test_single.py", line 14, in load_model
    tf.import_graph_def(od_graph_def, name='prefix')
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/util/deprecation.py", line 316, in new_func
    return func(*args, **kwargs)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/importer.py", line 554, in import_graph_def
    op_def=op_def)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 3160, in create_op
    op_def=op_def)
  File "/Users/zhangxin/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 1625, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): Number of ways to split should evenly divide the split dimension, but got split_dim 0 (size = 1) and num_split 2
	 [[Node: prefix/split = Split[T=DT_FLOAT, num_split=2, _device="/job:localhost/replica:0/task:0/device:CPU:0"](prefix/split/split_dim, _arg_prefix/input_images_0_0)]]
```