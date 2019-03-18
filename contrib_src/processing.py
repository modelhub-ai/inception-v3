from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json


class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            image = image.resize((299,299), resample = PIL.Image.LANCZOS)
            image = np.array(image).astype(np.float32)
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image

    def _preprocessAfterConversionToNumpy(self, npArr):
        npArr = self._preprocess_input(npArr)
        npArr = np.expand_dims(npArr, axis=0)
        return npArr

    def _preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def computeOutput(self, inferenceResults):
        probs = np.squeeze(np.asarray(inferenceResults))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result
