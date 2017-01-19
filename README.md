TensorFlow Graph Analsys
========================

# Receptive Field Size Analysis

This program automatically computes the receptive field size
of the top-most layer of the type Conv2D or PoolMax.
It is important that the receptive field size is big enough
to capture the characteristic structure of the underlying
image.

Usage:
```bash
rfsize.py MODEL
# there should be MODEL.meta, MODEL.index and MODEL.data_...
```

