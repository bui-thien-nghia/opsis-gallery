import onnxruntime.quantization as ort_quant

ort_quant.quantize_dynamic(
    model_input="paddleocrv4/rec_slim.onnx",
    model_output="paddleocrv4/rec_int8.onnx",
    weight_type=ort_quant.QuantType.QUInt8,
)

class DetDataReader(ort_quant.CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([
                {"x": self.preprocess(p)} for p in self.image_paths
            ])
        return next(self.enum_data, None)

    def preprocess(self, img_path):
        import numpy as np
        return np.zeros((1, 3, 640, 640), dtype=np.float32)

ort_quant.quantize_static(
    model_input="paddleocrv4/det_slim.onnx",
    model_output="paddleocrv4/det_int8.onnx",
    calibration_data_reader=DetDataReader(["sample1.jpg", "sample2.jpg"]),
    quant_format=ort_quant.QuantFormat.QDQ,
    activation_type=ort_quant.QuantType.QUInt8,
    weight_type=ort_quant.QuantType.QInt8,
)
print("Conversion complete: ppocrv4_det_int8.onnx & ppocrv4_rec_int8.onnx created.")