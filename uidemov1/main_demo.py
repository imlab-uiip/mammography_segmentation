
import warnings
warnings.filterwarnings("ignore")


import os
import sys
import glob
from PySide import QtGui
from PySide import QtCore
import gc


import numpy as np
import skimage.io as skio
import skimage.color as skcolor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from keras.models import load_model


import dialog_pyside as design

##############################################

from run00_common import buildProbMap, loadModelFromH5
###################

def convertMat2Pixmap(mat):
    if len(mat.shape)<3:
        mat=skcolor.gray2rgb(mat)
    elif mat.shape[2] ==1 :
        print("changed to rgb from tf gray shape")
        mat = mat[:,:,0]
        mat=skcolor.gray2rgb(mat)
    imgSiz=mat.shape
    ret=QtGui.QPixmap.fromImage(QtGui.QImage(mat.data, imgSiz[1], imgSiz[0], mat.strides[0], QtGui.QImage.Format_RGB888).copy())
    #ret=QtGui.QPixmap.fromImage(QtGui.QImage(mat.data, imgSiz[1], imgSiz[0], mat.strides[0], QtGui.QImage.Format_RGB16).copy())
    return ret

class DemoApp(QtGui.QDialog, design.Ui_Dialog):
    class DataSegmented:
        index = -1
        imgOrig = None
        lstImgSegm = None
        pxmOrig = None
        lstPxmSegm = None
        def __init__(self, index, imgOriginal, lstImgSegmented):
            self.index = index
            self.imgOrig = imgOriginal
            self.lstImgSegm = lstImgSegmented
            QtGui.QApplication.processEvents()
            self.pxmOrig = convertMat2Pixmap(self.imgOrig)
            QtGui.QApplication.processEvents()
            self.lstPxmSegm=[]
            for ii in self.lstImgSegm:
                self.lstPxmSegm.append(convertMat2Pixmap(ii))
                QtGui.QApplication.processEvents()
            QtGui.QApplication.processEvents()
    imgOrigin = None
    imgMask = None
    probMatrix = None
    listPathImages=None
    #listPathMaskes =None
    #pathModelIni=None
    #pathModelJson=None
    wdir = None
    listPixmaps = []
    batcher     = None
    modelDNN    = None
    dataSegmented = None
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.oldPath = os.path.expanduser('~')
        #
        self.scene = QtGui.QGraphicsScene()
        self.sceneSize = (40000, 40000)
        self.sceneRect = (-self.sceneSize[0] / 2.0, -self.sceneSize[1] / 2.0, self.sceneSize[0], self.sceneSize[1])
        self.sceneBnd = (self.sceneRect[0], self.sceneRect[1], self.sceneRect[0] + self.sceneSize[0],
                         self.sceneRect[1] + self.sceneSize[1])
        self.scene.setSceneRect(self.sceneRect[0], self.sceneRect[1], self.sceneRect[2], self.sceneRect[3])
        self.scene.addLine(self.sceneBnd[0], 0, self.sceneBnd[2], 0)
        self.scene.addLine(0, self.sceneBnd[1], 0, self.sceneBnd[3])
        self.graphicsView.setScene(self.scene)
        #
        self.cleanInfo()
        self.cleanSegmentedData()
        self.cleanModel()
        #
        self.horizontalSliderThreshold.setValue(90)
        self.spinBox.setValue(90)
        self.pushButtonPath.clicked.connect(self.slot_on_select_dialog)
        #self.pushButtonLoad.clicked.connect(self.slot_on_load_datadir)
        self.pushButtonTest.clicked.connect(self.slot_on_test)
        self.pushButtonPreview.clicked.connect(self.slot_on_preview_image)
        self.pushButtonSegm.clicked.connect(self.slot_on_segment_image)
        #
        self.comboBoxCls.activated.connect(self.slot_on_change_cls)
        self.checkBoxShowMask.clicked.connect(self.slot_on_change_cls)
        #
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

    def cleanInfo(self):
        self.imgOrigin = None
        self.imgMask = None
        self.probMatrix = None
        self.listPathImages = None
        #seld.listPathMaskes = None
        #self.pathModelIni   = None
        #self.pathModelJson  = None
        self.listWidgetImages.clear()
        self.comboBoxCls.clear()
    def cleanModel(self):
        self.batcher    = None
        self.modelDNN   = None
    def cleanSegmentedData(self):
        self.dataSegmented = None
    def isDataLoaded(self):
        return (self.wdir is not None) and (self.listPathImages is not None)# and (self.pathModelIni is not None) and (self.pathModelJson is not None)
    def loadModel(self,fpathname):
        if self.isDataLoaded():
            self.startProgress()
            QtGui.QApplication.processEvents()
            #self.batcher = BatcherImage2D(pathCSV=self.pathModelIni)
            QtGui.QApplication.processEvents()
            del self.modelDNN
            K.clear_session()
            gc.collect()
            self.modelDNN = loadModelFromH5(fpathname)
            QtGui.QApplication.processEvents()
            self.comboBoxCls.insertItems(0, ('clear','tumor'))
            self.stopProgress()
        else:
            print ('ERROR: demo-data is not loaded...')
    def loadDataDir(self,fpathname):
        self.cleanInfo()
        self.cleanSegmentedData()
        self.cleanModel()
        if self.wdir is not None:
            tisOk = True
            if os.path.isdir(self.wdir):# and os.path.isdir(dirModel):
                tlstImages = sorted(glob.glob('%s/*.png' % self.wdir))
                if len(tlstImages)>0:
                    self.listPathImages = [os.path.abspath(xx) for xx in tlstImages]
                else:
                    tisOk = False
            if tisOk:
                self.cleanScene()
                self.listWidgetImages.clear()
                for ii in self.listPathImages:
                    self.listWidgetImages.addItem(os.path.basename(ii))
                self.loadModel(fpathname)
            else:
                self.cleanInfo()
                self.showMessage("Incorrect Demo-Data directory [%s]" % self.wdir)
        else:
            self.showMessage('Please, select demo-data directory with <cfg.ini>')
    def slot_on_select_dialog(self):
        self.cleanInfo()
        #filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.oldPath, "Image-index (cfg.ini)")
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.oldPath)

        if isinstance(filename,tuple):
            filename = filename[0]
        filename = str(filename)
        if os.path.isfile(filename):
            self.wdir = os.path.abspath(os.path.dirname(filename))
            self.lineEdit.setText(filename)
            self.loadDataDir(filename)
        else:
            self.showMessage("Cant find <cfg.ini> [%s]" % filename)
    def startProgress(self):
        self.progressBar.setMaximum(0)
        QtGui.QApplication.processEvents()
    def stopProgress(self):
        self.progressBar.setMaximum(1)
        QtGui.QApplication.processEvents()
    def addPixmap2Scene(self, tpxm):
        tmpPixmap = self.scene.addPixmap(tpxm)
        tmpPixmap.setOffset(-tpxm.width() / 2, -tpxm.height() / 2)
        if self.listPixmaps:
            self.listPixmaps.append(tmpPixmap)
        else:
            self.listPixmaps = [tmpPixmap]
    def slot_on_preview_image(self):
        if self.isDataLoaded():
            tselectedIndex = self.listWidgetImages.currentRow()
            if tselectedIndex<0:
                self.showMessage('Please, select image in <Available images> list')
            else:
                print ('*** Selected index: %d' % tselectedIndex)
                self.cleanScene()
                self.startProgress()
                timg = skio.imread(self.listPathImages[tselectedIndex])
                print(self.wdir, os.path.basename(self.listPathImages[tselectedIndex]))
                try:

                    tmask = skio.imread(os.path.join(self.wdir,"maskes",os.path.basename(self.listPathImages[tselectedIndex])))
                except:
                    tmask = np.zeros_like(timg)
                    print("found no mask for such image")
                tpxm = convertMat2Pixmap(np.concatenate((timg,tmask),axis = 1))
                self.addPixmap2Scene(tpxm)
                self.stopProgress()
        else:
            self.showMessage('Demo-data is not loaded!')
    def cleanScene(self):
        for ii in self.listPixmaps:
            self.scene.removeItem(ii)
            ii = None
        self.listPixmaps = []
    def slot_on_test(self):
        if self.isDataLoaded():
            print ('row = %d' % self.listWidgetImages.currentRow())
    def slot_on_segment_image(self):
        if self.isDataLoaded():
            self.startProgress()
            QtGui.QApplication.processEvents()
            tselectedIndex = self.listWidgetImages.currentRow()
            print ('row = %d' % tselectedIndex)
            if tselectedIndex<0:
                self.showMessage('Please, select image in <Available images> list')
            else:
                if (self.dataSegmented is None) or (tselectedIndex != self.dataSegmented.index):

                    timgOriginal = skio.imread(self.listPathImages[tselectedIndex])
                    print (self.wdir, os.path.basename(self.listPathImages[tselectedIndex]))
                    try:

                        timgMask =skio.imread(os.path.join(self.wdir,"maskes",os.path.basename(self.listPathImages[tselectedIndex])))
                    except:
                        timgMask = np.zeros_like(timgOriginal)
                        print("found no mask for such image1")
                    if len(timgOriginal.shape) < 3:
                        timgOriginal = skcolor.gray2rgb(timgOriginal)
                        timgMask = skcolor.gray2rgb(timgMask)
                    self.imgOrigin = timgOriginal
                    self.imgMask = timgMask
                    self.modelDNN.save("tmp_model.h5")
                    del self.modelDNN
                    K.clear_session()
                    gc.collect()
                    self.modelDNN = load_model("tmp_model.h5")
                    tmodelFCNN = self.modelDNN
                    retProb, tmodelFCNN = buildProbMap( tmodelFCNN, timgOriginal)#2DO remove tmodel return
                    self.probMatrix = retProb
                    parThresh = float(self.horizontalSliderThreshold.value())/self.horizontalSliderThreshold.maximum()

                    retProbThresh = (255.0*(retProb>parThresh)).astype(np.uint8)
                    lstImgThresh=[]
                    for ii in range(retProb.shape[2]):
                        tmp = timgOriginal.copy()
                        tmpR = tmp[:,:,0]
                        tmpR[retProbThresh[:, :, ii] == 0] = 0
                        tmpR[retProbThresh[:, :, ii] >  0] = 255
                        tmp[:,:,0]=tmpR

                        tmpM = timgMask.copy()
                        tmpR = tmp[:, :, 0]
                        tmpR[retProbThresh[:, :, ii] == 0] = 0
                        tmpR[retProbThresh[:, :, ii] > 0] = 255
                        tmpM[:, :, 0] = tmpR

                        lstImgThresh.append(np.concatenate((tmp,tmpM),axis = 1))
                    self.dataSegmented = DemoApp.DataSegmented(tselectedIndex, timgOriginal, lstImgThresh)
                elif (self.dataSegmented is not None) and (tselectedIndex == self.dataSegmented.index):

                    timgOriginal = self.imgOrigin
                    timgMask = self.imgMask
                    retProb = self.probMatrix
                    parThresh = float(self.horizontalSliderThreshold.value())/self.horizontalSliderThreshold.maximum()

                    retProbThresh = (255.0*(retProb>parThresh)).astype(np.uint8)
                    lstImgThresh=[]
                    for ii in range(retProb.shape[2]):
                        tmp = timgOriginal.copy()
                        tmpR = tmp[:,:,0]
                        tmpR[retProbThresh[:, :, ii] == 0] = 0
                        tmpR[retProbThresh[:, :, ii] >  0] = 255
                        tmp[:,:,0]=tmpR

                        tmpM = timgMask.copy()
                        tmpR = tmp[:, :, 0]
                        tmpR[retProbThresh[:, :, ii] == 0] = 0
                        tmpR[retProbThresh[:, :, ii] > 0] = 255
                        tmpM[:, :, 0] = tmpR

                        lstImgThresh.append(np.concatenate((tmp, tmpM),axis = 1))
                    self.dataSegmented = DemoApp.DataSegmented(tselectedIndex, timgOriginal, lstImgThresh)

                self.drawSegmented()
            self.stopProgress()
        else:
            self.showMessage('Demo-data is not loaded!')
    def showMessage(self, txtMsg):
        msg = QtGui.QMessageBox()
        msg.setText(txtMsg)
        msg.exec_()
    def drawSegmented(self):
        if self.isDataLoaded() and (self.dataSegmented is not None):
            tidxCls = self.comboBoxCls.currentIndex()
            if tidxCls>-1:
                self.cleanScene()
                if self.checkBoxShowMask.isChecked():
                    self.addPixmap2Scene(self.dataSegmented.lstPxmSegm[tidxCls])

                else:
                    self.addPixmap2Scene(self.dataSegmented.pxmOrig)

    def slot_on_change_cls(self):
        self.drawSegmented()

if __name__=='__main__':
    app = QtGui.QApplication(sys.argv)
    form = DemoApp()
    form.show()
    app.exec_()
