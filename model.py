import kfserving
from typing import List, Dict
from PIL import Image
import base64
import io
import os
import logging
import json
import SimpleITK as sitk
import numpy as np

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()
    
filename = 'temp.jpg'

model_name = os.getenv('MODEL_NAME',None)


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.ready = True

    def predict(self, inputs: Dict) -> Dict:
        del inputs['instances']
        logging.info("prep =======> %s",str(type(inputs)))
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        data = json_data["signatures"]["inputs"][0][0]["data"]
        #data1 = json_data["signatures"]["inputs"][0][0]["data1"]
        #data2 = json_data["signatures"]["inputs"][0][0]["data2"]
        #writing the inp image
        #b64_filewriter(filename, data)
        ct_scan, origin, spacing = load_itk('images/original_sub.mhd')
        
        with open('images/AMRD14-segmentation.jpeg', 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = base64.b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        return {"out_image":base64_string}

if __name__ == "__main__":
    model = KFServingSampleModel(model_name)
    model.load()
    kfserving.KFServer(workers=1).start([model])
