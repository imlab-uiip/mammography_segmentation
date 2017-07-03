from PySide import QtCore, QtGui

#from main_demo import DemoApp
#
class MyView(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super(MyView, self).__init__()
        self.name = "MyView"

    def wheelEvent(self, ev):
        # print ('----')
        scaleFactor = 1.1
        if ev.delta() > 0:
            self.scale(scaleFactor, scaleFactor)
        else:
            self.scale(1. / scaleFactor, 1. / scaleFactor)