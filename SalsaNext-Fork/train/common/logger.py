"""TensorBoard logging helpers used by the training loop."""

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Compatibility wrapper around PyTorch's TensorBoard writer.

    The original project used TensorFlow 1.x summary protobufs. Keeping the
    wrapper's public method names avoids changing the training code while
    removing the TensorFlow 1.x and deprecated SciPy dependencies.
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        if hasattr(value, "item"):
            value = value.item()
        self.writer.add_scalar(tag, float(value), step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        for index, image in enumerate(images):
            image = np.asarray(image)
            if image.ndim == 2:
                dataformats = "HW"
            elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                dataformats = "HWC"
            else:
                raise ValueError("Images must have shape [H,W] or [H,W,C]")
            self.writer.add_image(
                "{}/{}".format(tag, index), image, step, dataformats=dataformats
            )
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        self.writer.add_histogram(tag, np.asarray(values), step, bins=bins)
        self.writer.flush()

    def close(self):
        self.writer.close()
