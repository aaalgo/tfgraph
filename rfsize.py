#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
import argparse
from toposort import toposort
import tensorflow as tf

class TFGraph(object):
    def __init__ (self, model):
        graph = tf.Graph()
        params = {}
        deps = {}
        with graph.as_default():
            saver = tf.train.import_meta_graph(model + '.meta')
            graph_def = graph.as_graph_def()
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True

            with tf.Session(config=config) as sess:
                sess.run(init)
                saver.restore(sess, model)
                # scan the nodes
                for node in graph_def.node:
                    name = node.name
                    deps[name] = set(node.input)
                    if 'Backprop' in name:
                        # skip backprop ones
                        continue
                    if 'gradients' in name:
                        continue
                    attr = {}
                    for key in ['padding', 'ksize', 'strides']:
                        if key in node.attr:
                            attr[key] = node.attr[key]
                    if (not ('ksize' in attr)) and len(attr) == 2:
                        # it's likely we have ve hit a Conv2D op, try to find ksize by
                        # actually loading the weighting
                        if len(node.input)  < 2 or not ('weights' in node.input[1]):
                            print(node.name)
                            print(node.op)
                            print(node.input)
                            raise Exception('xx')
                        W_name = node.input[1] + ':0'
                        W = graph.get_tensor_by_name(W_name)
                        w, = sess.run([W])
                        K1, K2, _, _ = w.shape
                        attr['ksize'] = [1, K1, K2, 1]
                    elif 'ksize' in attr:
                        attr['ksize'] = [int(i) for i in attr['ksize'].list.i]

                    if 'padding' in attr:
                        attr['padding'] = attr['padding'].s
                    if 'strides' in attr:
                        attr['strides'] = [int(i) for i in attr['strides'].list.i]
                    if not (len(attr) == 0 or len(attr) == 3):
                        print("node %s of type %s is funny, we cannot support it!" % (node.name, node.op))
                        print(attr)
                        sys.exit(1)
                        pass
                    if len(attr) != 3:
                        continue

                    attr['op'] = node.op
                    params[name] = attr
                    pass
                pass
            pass
        self.params = params
        self.deps = deps
        pass

    def sort_convlike (self):
        v = []
        for group in toposort(self.deps):
            group = [x for x in group if x in self.params]
            if len(group) > 2:
                logging.error('non-serial graph, groups=%s' % group)
                raise Except()
            if len(group) == 0:
                continue
            name = group[0]
            param = self.params[name]
            #print('%s: %s' % (name, param['op']))
            ksize = param['ksize']
            strides = param['strides']
            padding = param['padding']
            #print('\tksize=%s, strides=%s, padding=%s' % (ksize, strides, padding))
            v.append((name, param['op'], ksize, strides, padding))
        return v
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--model', default='model/20000', help='model')
    args = parser.parse_args()
    graph = TFGraph(args.model)
    ops = graph.sort_convlike()
    ops.reverse()
    rf = 1
    not_started = True
    for name, op, ksize, strides, padding in ops:
        assert op in ['Conv2DBackpropInput', 'Conv2D', 'MaxPool']
        if op == 'Conv2DBackpropInput':
            assert not_started
            continue
        print('%s %s %s %s %s => %d' % ( name, op, ksize, strides, padding, rf))
        not_started = False
        _, K1, K2, _ = ksize
        _, S1, S2, _ = strides
        assert K1 == K2
        assert S2 == S2
        rf = (rf - 1) * S1 + K1
        pass
    print("Receptive field: %d" % rf)





