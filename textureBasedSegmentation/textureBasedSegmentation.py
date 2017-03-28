import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import sys
import pickle
import numpy as np

#
# textureBasedSegmentation
#

class textureBasedSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "textureBasedSegmentation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This module renders 3d textured models, and enables the segmentation of the loaded textured model based on
    texture (RGB, HSL) value based machine learning classification techniques
    """
    self.parent.acknowledgementText = """
    This module was created by Brandon Chan, Nuwan Perera, and Mareena Mallory
    """ 

#
# textureBasedSegmentationWidget
#

class textureBasedSegmentationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Model: ", self.inputSelector)

    #
    # input texture selector
  	#
    self.inputTextureSelector = slicer.qMRMLNodeComboBox()
    self.inputTextureSelector.nodeTypes = ( ("vtkMRMLVectorVolumeNode"), "" )
    self.inputTextureSelector.selectNodeUponCreation = True
    self.inputTextureSelector.addEnabled = False
    self.inputTextureSelector.removeEnabled = False
    self.inputTextureSelector.noneEnabled = False
    self.inputTextureSelector.showHidden = False
    self.inputTextureSelector.showChildNodeTypes = False
    self.inputTextureSelector.setMRMLScene( slicer.mrmlScene )
    self.inputTextureSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Texture: ", self.inputTextureSelector)	
	
    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # Red threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 1
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 255
    self.imageThresholdSliderWidget.value = 0
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    #parametersFormLayout.addRow("Red value", self.imageThresholdSliderWidget)

    #
    # Green threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 1
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 255
    self.imageThresholdSliderWidget.value = 0
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    #parametersFormLayout.addRow("Green value", self.imageThresholdSliderWidget)
	
    #
    # Blue threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 1
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 255
    self.imageThresholdSliderWidget.value = 0
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    #parametersFormLayout.addRow("Blue value", self.imageThresholdSliderWidget)
	
    #
    # +/- threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 1
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 255
    self.imageThresholdSliderWidget.value = 0
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    #parametersFormLayout.addRow("+/- Threshold", self.imageThresholdSliderWidget)
	
    #
    # Texture Selection
    #
    self.textureSelector = qt.QComboBox()
    parametersFormLayout.addRow("Texture:", self.textureSelector)
    self.textureSelector.addItem('Bone')
    self.textureSelector.addItem('Muscle')
    self.textureSelector.addItem('Tendon')
    self.textureSelector.addItem('Cartilage')

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply Texture")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)
	
    #
    # Export Triangles Button
    #
    self.segmentButton = qt.QPushButton("Export Triangles")
    self.segmentButton.toolTip = "Export data to be segmented"
    self.segmentButton.enabled = False
    parametersFormLayout.addRow(self.segmentButton)
	
    #
    # Import Segmentation Data
    #
    self.importButton = qt.QPushButton("Import Segmented Data")
    self.importButton.toolTip = "Load in segmented data"
    self.importButton.enabled = False
    parametersFormLayout.addRow(self.importButton)
	
    #
    # Surface Area Display 
    #
    self.surfaceAreaDisplay = qt.QLineEdit("Surface Area: ")
    parametersFormLayout.addRow(self.surfaceAreaDisplay)
	
    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.segmentButton.connect('clicked(bool)',self.onSegmentButton)
    self.importButton.connect('clicked(bool)',self.onImportButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputTextureSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.textureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.inputTextureSelector.currentNode() # and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = textureBasedSegmentationLogic()
    logic.ShowTextureOnModel(self.inputSelector.currentNode(), self.inputTextureSelector.currentNode())
    logic.MapRGBtoPoints(self.inputSelector.currentNode(), self.inputTextureSelector.currentNode())
    self.surfaceAreaDisplay.setText('Surface Area: ' + str(logic.GetSurfaceArea(self.inputSelector.currentNode())) + ' mm^2')
    
    self.segmentButton.enabled = True
    self.importButton.enabled = True
    #logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold)

  def onSegmentButton(self):
    logic = textureBasedSegmentationLogic()
    logic.SegmentTriangles(self.inputSelector.currentNode(), self.inputTextureSelector.currentNode())
    #self.importSegmentationButton.enabled = True
	
  def onImportButton(self):
    logic = textureBasedSegmentationLogic()
    logic.renderSegmentedData(self.inputSelector.currentNode(), self.outputSelector.currentNode(), self.textureSelector.currentIndex)
	
#
# textureBasedSegmentationLogic
#

class textureBasedSegmentationLogic(ScriptedLoadableModuleLogic):
  #
  # Renders and displays tetured model in the slicer scene
  #  
  def ShowTextureOnModel(self, modelNode, textureImageNode):
    modelDisplayNode=modelNode.GetDisplayNode()
    modelDisplayNode.SetBackfaceCulling(0)
    textureImageFlipVert=vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
    modelDisplayNode.SetTextureImageDataConnection(textureImageFlipVert.GetOutputPort())

  #
  # Gets the surface area of the model selected
  #
  def GetSurfaceArea(self,modelNode):
    massProperties = vtk.vtkMassProperties()
    triangleFilter = vtk.vtkTriangleFilter()
    massProperties.SetInputConnection(triangleFilter.GetOutputPort())
    triangleFilter.SetInputData(modelNode.GetPolyData())
    surfaceArea = massProperties.GetSurfaceArea()
    return surfaceArea
	
  #
  # Function that segments the model through a trained machine learning model 
  #
  def SegmentTriangles(self, modelNode, outputModelNode):
    # Get triangle data and point data from the original (unsegmented model)
    originalPolyData = modelNode.GetPolyData()
    originalPointData = originalPolyData.GetPointData()
	
	# Get rgb values of each point in the model and store in arrays
    redValues = originalPointData.GetArray('ColorRed')
    greenValues = originalPointData.GetArray('ColorGreen')
    blueValues = originalPointData.GetArray('ColorBlue')
    
	# Get number of entries in the array (number of points in the model)
    lengthTuples = int(redValues.GetNumberOfTuples()) 
    selectedPointIds = vtk.vtkIdTypeArray()
	
    outputArray = np.zeros((lengthTuples,4))
	
	# Covert data to a numpy array
    for i in range(lengthTuples):
      outputArray[i][0] = i
      outputArray[i][1] = redValues.GetValue(i)
      outputArray[i][2] = greenValues.GetValue(i)
      outputArray[i][2] = blueValues.GetValue(i)
	
    #np.savetxt('C:/Users/Brand/OneDrive/Documents/outputArray.csv',outputArray,delimiter=',')
	
    outputArray.dump('C:/Users/Brand/OneDrive/Documents/outputArray.pkl')
	
    #with open('C:/Users/Brand/OneDrive/Documents/outputArray.pkl', 'wb') as f:
    #  pickle.dump(outputArray, f)
    # pickle.dump(outputArray, open(os.chdir('C:/Users/Brand/OneDrive/Documents/outputArray.pkl'),'wb'))
    #serializedArray = pickle.dumps(outputArray, protocol=0)
	
    return 0
	
  def renderSegmentedData(self, modelNode, outputModelNode, selectedTexture):
	
    fullPolyData = modelNode.GetPolyData()
    pointData=fullPolyData.GetPointData()
	
    data = np.load('C:/Users/Brand/OneDrive/Documents/CISC 472/test segmentation_model/classified_texture.pkl')
    size_of_data = data.shape
    print(size_of_data)
    print(int(data[1][0]))
    print(' ')
    print(selectedTexture)
	
    segmentedPointIds = vtk.vtkIdTypeArray()

    print('begin classifiacation')
    for point in range(size_of_data[1]):
      if int(data[1][point]) == selectedTexture:
        segmentedPointIds.InsertNextValue(int(data[0][point]))
        segmentedPointIds.InsertNextValue(int(data[0][point]))
		
    print('calssification done')
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(segmentedPointIds)
    selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1);

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0,fullPolyData)
    extractSelection.SetInputData(1,selection);
    extractSelection.Update();
	
    convertToPolydata = vtk.vtkDataSetSurfaceFilter()
    convertToPolydata.SetInputConnection(extractSelection.GetOutputPort())
    convertToPolydata.Update()
    outputModelNode.SetAndObservePolyData(convertToPolydata.GetOutput())

    if not outputModelNode.GetDisplayNode():
      md2 = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode(md2)
      outputModelNode.SetAndObserveDisplayNodeID(md2.GetID()) 
	
    print('done?')
    return 0	
	
  #
  # Add texture data to scalars
  #
  def MapRGBtoPoints(self, modelNode, textureImageNode):
	# Get triangles/verticies data from model 
    polyData = modelNode.GetPolyData()
	
    textureImageFlipVert = vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
    textureImageFlipVert.Update() 
    textureImageData = textureImageFlipVert.GetOutput()
	
	# Get location of points that make up the model and associated texture coordinates
    pointData = polyData.GetPointData()
    tcoords = pointData.GetTCoords()
    numOfPoints = pointData.GetNumberOfTuples()
	
	# Ensure number of texture coordinates match the number of vertecies in the model 
    assert numOfPoints == tcoords.GetNumberOfTuples(), "Number of texture coordinates does not equal number of points"
    textureSamplingPointsUv = vtk.vtkPoints()
    textureSamplingPointsUv.SetNumberOfPoints(numOfPoints)
	
	# Get u,v texture coordinates from model node
    for pointIndex in xrange(numOfPoints):
      uv = tcoords.GetTuple2(pointIndex)
      textureSamplingPointsUv.SetPoint(pointIndex, uv[0], uv[1], 0)

	# Calculate transforms for loaded texture image
    textureSamplingPointDataUv = vtk.vtkPolyData()
    uvToXyz=vtk.vtkTransform()
    textureImageDataSpacingSpacing = textureImageData.GetSpacing()
    textureImageDataSpacingOrigin = textureImageData.GetOrigin()
    textureImageDataSpacingDimensions = textureImageData.GetDimensions()
    uvToXyz.Scale(textureImageDataSpacingDimensions[0]/textureImageDataSpacingSpacing[0], textureImageDataSpacingDimensions[1]/textureImageDataSpacingSpacing[1], 1)
    uvToXyz.Translate(textureImageDataSpacingOrigin)
    textureSamplingPointDataUv.SetPoints(textureSamplingPointsUv)
    transformPolyDataToXyz = vtk.vtkTransformPolyDataFilter()
    transformPolyDataToXyz.SetInputData(textureSamplingPointDataUv)
    transformPolyDataToXyz.SetTransform(uvToXyz)
	
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetInputConnection(transformPolyDataToXyz.GetOutputPort())
    probeFilter.SetSourceData(textureImageData)
    probeFilter.Update()
    rgbPoints = probeFilter.GetOutput().GetPointData().GetArray('ImageScalars')
	
	# Initialize arrays to store rgb values of the pixel mapped to a given vertex point in the model
    colorArrayRed = vtk.vtkDoubleArray()
    colorArrayRed.SetName('ColorRed')
    colorArrayRed.SetNumberOfTuples(numOfPoints)
    colorArrayGreen = vtk.vtkDoubleArray()
    colorArrayGreen.SetName('ColorGreen')
    colorArrayGreen.SetNumberOfTuples(numOfPoints)
    colorArrayBlue = vtk.vtkDoubleArray()
    colorArrayBlue.SetName('ColorBlue')
    colorArrayBlue.SetNumberOfTuples(numOfPoints)
	
	# Iterate through each point and get / store the rgb value of the pixel it is mapped to
    for pointIndex in xrange(numOfPoints):
      rgb = rgbPoints.GetTuple3(pointIndex)
      colorArrayRed.SetValue(pointIndex,rgb[0])
      colorArrayGreen.SetValue(pointIndex,rgb[1])
      colorArrayBlue.SetValue(pointIndex,rgb[2])
	
    colorArrayRed.Modified()
    colorArrayGreen.Modified()
    colorArrayBlue.Modified()
	
    pointData.AddArray(colorArrayRed)
    pointData.AddArray(colorArrayGreen)
    pointData.AddArray(colorArrayBlue)
	
    pointData.Modified()
    polyData.Modified()
    return


  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qpixMap = qt.QPixmap().grabWidget(widget)
    qimage = qpixMap.toImage()
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('textureBasedSegmentationTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class textureBasedSegmentationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_textureBasedSegmentation1()

  def test_textureBasedSegmentation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = textureBasedSegmentationLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
