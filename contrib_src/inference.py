import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
from inceptionV3 import InceptionV3
from keras.applications.imagenet_utils import decode_predictions

class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model
        self._model = InceptionV3()
        self._model.load_weights('model/model.h5')
        self._model._make_predict_function()

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with caffe2
        results = self._model.predict(inputAsNpArr)
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
