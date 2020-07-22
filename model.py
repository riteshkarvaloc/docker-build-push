import kfserving
from typing import List, Dict
from PIL import Image
import base64
import io
import os
import logging
import json


filename = 'temp.jpg'

def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()


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
        #writing the inp image
        #b64_filewriter(filename, data)
        
        with open('images/AMRD14-segmentation.jpeg', 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = base64.b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        return {'out_image':base64_string}

if __name__ == "__main__":
    model = KFServingSampleModel("kfserving-custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])
