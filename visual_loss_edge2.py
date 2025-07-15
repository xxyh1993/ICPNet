import visdom
# import time
# import numpy as np


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.vis.line([2.0], [0.], win='loss', opts=dict(title='loss', legend=['train_loss']))
        self.vis.line([5500.], [0.], win='edge', opts=dict(title='edge_loss', legend=['edge_loss']))
        self.vis.line([[0., 0.]], [0.], win='top1_acc.', opts=dict(title='top1_acc.', legend=['train@top1_acc', 'test@top1_acc']))
        self.vis.line([0.0], [0.], win='lr', opts=dict(title='lr', legend=['lr']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i4@top1', opts=dict(title='AG_i4@top1', legend=['hor', 'ul', 'ur', 'ver']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i6@top1', opts=dict(title='AG_i6@top1', legend=['hor', 'ul', 'ur', 'ver']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i8@top1', opts=dict(title='AG_i8@top1', legend=['hor', 'ul', 'ur', 'ver']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i10@top1', opts=dict(title='AG_i10@top1', legend=['hor', 'ul', 'ur', 'ver']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i12@top1', opts=dict(title='AG_i12@top1', legend=['hor', 'ul', 'ur', 'ver']))
        # self.vis.line([[0., 0., 0., 0.]], [0.], win='AG_i14@top1', opts=dict(title='AG_i14@top1', legend=['hor', 'ul', 'ur', 'ver']))

    def plot_loss(self, d, iterations):
        self.vis.line([d], [iterations], win='loss', update='append')

    def plot_edge_loss(self, d, iterations):
        self.vis.line([d], [iterations], win='edge', update='append')

    def plot_acc_stack(self, d, iterations):
        self.vis.line([[d[0], d[1]]], [iterations], win='top1_acc.', update='append')

    def plot_lr(self, d, epoch):
        self.vis.line([d], [epoch], win='lr', update='append')

    def plot_i4(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i4@top1', update='append')

    def plot_i6(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i6@top1', update='append')

    def plot_i8(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i8@top1', update='append')

    def plot_i10(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i10@top1', update='append')

    def plot_i12(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i12@top1', update='append')

    def plot_i14(self, d, epoch):
        self.vis.line([[d[0], d[1], d[2], d[3]]], [epoch], win='AG_i14@top1', update='append')
