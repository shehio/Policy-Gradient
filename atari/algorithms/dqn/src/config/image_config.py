class ImageConfig:
    def __init__(self, target_h: int = 80,
                 target_w: int = 64,
                 crop_top: int = 20):
        self.target_h = target_h
        self.target_w = target_w
        self.crop_top = crop_top 