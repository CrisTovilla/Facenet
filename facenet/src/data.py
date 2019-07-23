class DataSerializer:
    def __init__(self, msg, detected_name, detected_pos, detected_precision):        
        self.msg = msg
        self.detected_name = detected_name
        self.detected_pos = detected_pos
        self.detected_precision = detected_precision