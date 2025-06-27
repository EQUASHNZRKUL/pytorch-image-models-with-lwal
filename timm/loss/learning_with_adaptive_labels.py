""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

LAST_Z_OF_LABEL = torch.tensor([[ 6.2812e+00, -3.5781e+00,  8.3125e+00, -7.6562e-01, -4.3125e+00,
         -2.0156e+00,  5.3750e+00, -3.4844e+00, -1.5703e+00, -4.1562e+00,
         -6.6528e-03,  5.7373e-03, -3.9673e-03,  7.2632e-03, -3.2654e-03,
         -1.8921e-02,  1.4709e-02, -3.0518e-03,  3.4637e-03, -1.6602e-02,
          6.6528e-03, -2.3193e-02, -1.5625e-02,  1.9302e-03, -4.3030e-03,
          8.3008e-03,  7.6294e-03, -1.1597e-02,  8.4839e-03,  6.9885e-03,
          1.5259e-02,  1.6479e-03,  6.7444e-03,  5.3101e-03, -5.6152e-03,
         -5.9814e-03, -7.9956e-03,  1.3306e-02, -1.2634e-02, -7.9346e-03,
          4.5166e-03,  1.6602e-02,  1.6846e-02,  8.2397e-03, -5.6152e-03,
          2.4414e-03,  1.1841e-02,  3.2959e-03, -5.2490e-03,  6.9580e-03,
         -2.1973e-03, -9.6436e-03,  5.4932e-03,  7.3242e-04, -7.4768e-03,
         -2.0386e-02, -6.2256e-03, -7.2937e-03, -3.4790e-03, -6.0425e-03,
          3.8757e-03, -4.2114e-03,  9.8877e-03,  1.8311e-04,  1.0986e-03,
          1.0986e-02, -1.3062e-02,  8.0566e-03,  1.1475e-02, -1.2360e-03,
          5.6763e-03, -2.3193e-03,  1.6235e-02,  4.9133e-03, -2.9053e-02,
         -6.1035e-04,  2.1973e-03,  1.1230e-02, -1.6724e-02, -9.0027e-04,
         -1.9409e-02,  1.7944e-02, -2.5269e-02, -5.4321e-03,  8.4229e-03,
         -1.3611e-02, -1.8311e-03,  3.8452e-03, -3.0975e-03,  9.3384e-03,
          6.9809e-04,  5.9814e-03, -8.5449e-03, -1.6357e-02, -7.8735e-03,
         -1.6968e-02,  8.6060e-03,  9.7656e-03, -3.9978e-03, -1.7700e-03],
        [ 1.2000e+01, -2.0625e+00,  1.1875e+00, -2.0312e+00, -3.1250e+00,
         -3.3750e+00,  5.1562e-01, -2.1875e+00,  5.2188e+00, -5.7812e+00,
         -3.3203e-02,  1.5869e-02, -8.1787e-03, -3.2349e-03,  3.3203e-02,
         -1.0254e-02,  1.5625e-02, -2.1240e-02,  1.1108e-02, -9.4604e-03,
         -1.7212e-02,  1.0757e-03, -2.0142e-02,  1.6846e-02,  2.4414e-03,
          8.6670e-03,  1.7090e-03, -8.4229e-03, -3.9307e-02,  6.2561e-03,
          1.1230e-02, -1.2573e-02, -1.6235e-02, -6.4697e-03,  1.2512e-02,
         -1.3916e-02,  7.3242e-04,  8.4229e-03,  4.8218e-03,  5.7373e-03,
          1.7456e-02,  1.5869e-02,  1.3672e-02, -3.9673e-03,  2.4414e-04,
          1.5259e-02,  3.8147e-04,  4.0527e-02, -8.3008e-03,  1.2207e-03,
         -3.2959e-03, -1.3428e-03,  9.8877e-03, -5.9814e-03, -1.0681e-02,
          2.3438e-02, -3.3203e-02,  5.8594e-03, -5.1270e-03,  1.4038e-02,
         -4.5471e-03,  1.0376e-02,  3.4485e-03, -1.3306e-02,  1.7212e-02,
          2.9663e-02,  1.8921e-03,  2.1362e-04,  3.9795e-02, -9.6436e-03,
          7.3853e-03,  1.9531e-03,  4.5471e-03, -3.0273e-02,  1.0681e-02,
         -1.1963e-02, -7.2021e-03,  1.4893e-02,  7.6599e-03, -2.7344e-02,
         -1.8433e-02,  7.9346e-03,  2.5635e-02,  4.9438e-03,  2.2217e-02,
         -3.3203e-02, -1.0315e-02,  1.5869e-02, -3.6316e-03, -2.1729e-02,
          1.6235e-02,  5.4169e-04, -1.3428e-03,  7.5684e-03,  3.3112e-03,
         -2.7466e-04,  4.0283e-03,  1.0742e-02,  2.3193e-03, -7.1411e-03],
        [ 1.7812e+00,  3.7500e+00, -2.7539e-01, -1.2734e+00, -8.0859e-01,
         -1.6719e+00, -1.7734e+00, -1.1172e+00,  6.0156e-01,  6.5625e-01,
         -1.0803e-02,  1.8433e-02, -5.9204e-03, -5.5542e-03, -1.1902e-03,
          1.0376e-03,  3.2043e-03, -4.0283e-03, -4.3640e-03,  4.4556e-03,
          1.1841e-02, -1.1047e-02, -3.1738e-03, -1.4709e-02,  6.6528e-03,
          9.1553e-03,  2.2583e-03, -6.7139e-04, -7.8735e-03,  1.1230e-02,
          3.8452e-03,  1.7334e-02, -4.6997e-03,  8.7891e-03,  9.2773e-03,
         -5.1270e-03, -4.2725e-04, -3.3264e-03, -4.2725e-03, -7.9346e-03,
         -3.7231e-03,  2.3956e-03,  7.5989e-03, -4.4556e-03,  8.1787e-03,
          3.3569e-03,  6.9580e-03, -1.0010e-02, -1.1719e-02, -6.1035e-04,
          1.6846e-02, -1.0132e-02, -5.2490e-03,  2.8076e-03,  1.0071e-03,
         -6.8665e-03, -3.2959e-03, -1.6113e-02,  1.9836e-03,  2.4719e-03,
         -3.1433e-03,  6.7139e-03,  1.6235e-02,  1.4160e-02,  1.9531e-03,
          2.2583e-03,  1.5259e-03, -1.4343e-03,  7.4158e-03, -6.1646e-03,
          5.2490e-03, -6.1035e-03,  1.7548e-03,  1.6846e-02,  6.7139e-03,
         -4.4556e-03, -1.0803e-02, -1.7090e-03, -5.6458e-03,  1.1597e-02,
          1.8433e-02,  1.2695e-02,  1.5137e-02, -2.0630e-02, -8.5449e-04,
         -1.0437e-02,  6.9885e-03,  7.4768e-03,  3.9368e-03,  8.7891e-03,
         -4.1199e-03, -2.0630e-02,  0.0000e+00,  9.3994e-03, -8.3008e-03,
         -1.0254e-02,  1.4832e-02,  8.3008e-03, -1.4221e-02, -3.6621e-04],
        [-4.4688e+00, -5.5312e+00,  3.3750e+00,  7.7500e+00, -5.4297e-01,
          6.8750e+00,  1.9766e+00,  3.2812e-01, -4.1250e+00, -5.8125e+00,
          7.1716e-03, -7.5378e-03,  7.9956e-03, -4.2725e-03,  2.1484e-02,
         -2.1118e-02, -3.2227e-02, -8.6670e-03, -3.4485e-03, -1.3428e-03,
         -7.6294e-04, -2.6489e-02, -2.1973e-03,  2.4109e-03, -3.3569e-04,
         -1.4832e-02,  1.8799e-02,  3.5889e-02, -7.6294e-04,  6.5002e-03,
          8.0566e-03, -4.8828e-03,  5.1880e-04, -2.8687e-03,  6.1035e-05,
         -1.4771e-02, -1.1047e-02,  1.5869e-03, -2.0264e-02, -1.0864e-02,
         -5.1880e-03,  4.1809e-03, -2.1362e-02,  8.7891e-03, -1.1719e-02,
          8.9722e-03,  4.8523e-03, -3.1128e-03,  1.7090e-03,  2.5513e-02,
         -1.3062e-02, -1.1780e-02, -2.5635e-03,  2.9297e-03,  2.0599e-03,
         -2.2949e-02, -1.6113e-02, -2.3193e-03, -8.7891e-03, -5.7373e-03,
          8.5449e-03, -7.8125e-03,  2.3438e-02, -3.8574e-02,  1.0986e-03,
          1.1536e-02, -2.4109e-03, -1.0376e-02,  1.5625e-02,  1.1230e-02,
         -6.6528e-03, -7.8125e-03,  1.9531e-03, -3.1006e-02, -1.4954e-02,
          3.0518e-03,  1.0010e-02, -7.2937e-03,  3.9368e-03,  4.4556e-03,
         -1.2573e-02,  2.8381e-03,  1.7395e-03,  4.6387e-03, -9.7656e-03,
          2.2705e-02, -1.2390e-02,  1.2451e-02,  1.2207e-02, -8.7891e-03,
          1.3245e-02,  8.1787e-03, -9.2773e-03,  5.0049e-03,  3.1281e-03,
          1.2573e-02, -3.3875e-03,  1.1230e-02,  9.7656e-04, -1.2146e-02],
        [ 4.0312e+00, -1.6797e+00,  1.3047e+00, -1.4688e+00, -1.5469e+00,
         -1.9297e+00,  1.5859e+00, -2.2812e+00,  2.9219e+00, -9.4922e-01,
         -6.7139e-03,  1.9043e-02, -1.1230e-02, -1.4038e-03, -1.2634e-02,
         -3.1738e-03, -8.6060e-03,  7.0190e-03, -4.1809e-03,  2.4414e-03,
          1.3000e-02, -5.4932e-03, -1.2329e-02, -1.0803e-02,  1.6968e-02,
          4.1199e-03, -5.8899e-03, -5.9204e-03, -6.5613e-03, -1.0437e-02,
          8.9111e-03,  7.6904e-03, -1.0193e-02,  5.9509e-03,  6.1951e-03,
          2.0752e-03,  3.0029e-02, -7.9346e-03, -4.9438e-03, -2.8687e-03,
          8.3618e-03, -7.4768e-03,  7.2937e-03, -6.2866e-03,  7.3242e-03,
          9.3384e-03,  7.0190e-03,  2.0142e-03,  3.7231e-03,  8.5449e-03,
          1.3794e-02, -8.5449e-03, -1.2939e-02, -1.1108e-02, -1.1597e-02,
         -7.2632e-03,  4.5166e-03, -7.6904e-03,  1.2695e-02, -8.5449e-03,
         -1.0223e-03, -4.5013e-04,  7.9956e-03, -6.1035e-03,  1.4648e-03,
          4.2725e-04, -3.8147e-03,  4.1809e-03, -8.5449e-04, -9.2773e-03,
         -1.5259e-03, -6.3477e-03,  1.3489e-02, -6.4087e-03,  3.2654e-03,
          9.8267e-03,  2.1362e-03,  3.0518e-05, -9.5215e-03, -2.9449e-03,
         -8.1177e-03,  4.5166e-03,  5.4016e-03,  1.2573e-02, -1.2207e-03,
          1.7822e-02, -3.7842e-03,  7.2327e-03, -7.8735e-03, -2.2888e-03,
         -3.7537e-03,  2.2888e-03, -1.3184e-02,  1.0498e-02,  8.5449e-03,
          4.2419e-03,  1.2817e-02,  4.5471e-03, -2.2583e-03,  1.2268e-02],
        [-1.8516e+00, -4.4062e+00, -2.1875e-01,  3.2969e+00,  8.2422e-01,
          7.5625e+00, -3.3281e+00,  4.4688e+00, -3.9688e+00, -2.1875e+00,
         -1.1597e-03,  1.3672e-02,  1.5076e-02, -5.2490e-03, -4.7607e-03,
         -1.3672e-02,  4.3030e-03, -7.6904e-03,  1.9775e-02, -1.6846e-02,
          2.8381e-03, -4.6997e-03, -3.1738e-03, -9.2163e-03, -1.0620e-02,
          4.6082e-03,  6.2256e-03, -2.4719e-03,  2.1362e-03,  4.9133e-03,
          2.3804e-03, -4.7607e-03,  1.0864e-02,  9.4604e-03,  3.0518e-04,
         -3.0518e-04, -6.2866e-03, -2.1362e-04, -5.8594e-03, -7.4463e-03,
          7.3547e-03, -9.0790e-04, -1.2817e-02,  1.4648e-02, -1.4038e-02,
         -6.0425e-03, -3.3569e-04, -2.5635e-03,  2.1118e-02,  8.9111e-03,
         -2.3193e-03, -8.3618e-03,  1.5015e-02, -5.7373e-03, -2.7954e-02,
         -1.3306e-02,  1.5869e-03,  1.1169e-02, -1.1475e-02,  5.6152e-03,
          6.1951e-03,  1.1780e-02,  1.3489e-02, -2.6978e-02, -1.3550e-02,
         -4.3945e-03,  1.4801e-03,  1.7090e-02,  2.6123e-02,  2.3499e-03,
         -1.0010e-02, -9.3994e-03,  1.4526e-02, -1.2390e-02,  1.2878e-02,
          1.1353e-02, -1.0132e-02, -1.3062e-02, -2.3315e-02, -1.0742e-02,
         -1.0986e-02, -1.6785e-03, -1.1139e-03, -1.2054e-03, -5.4932e-03,
          6.4697e-03,  4.1504e-03,  1.7853e-03,  1.2360e-03,  2.8687e-03,
          1.6602e-02,  6.4087e-03, -1.6479e-02,  2.4902e-02, -7.8125e-03,
          4.7607e-03, -1.7090e-02, -8.5449e-04,  5.1880e-03, -4.6387e-03],
        [ 9.1016e-01,  1.7676e-01, -1.5234e+00,  2.4805e-01, -5.6641e-02,
         -3.4219e+00,  6.5000e+00, -3.4375e+00,  3.6562e+00, -3.1094e+00,
         -5.6763e-03,  3.1738e-02,  2.6245e-03,  7.3547e-03,  1.4343e-03,
         -1.5503e-02, -4.6997e-03, -1.0376e-02,  1.5381e-02, -1.8555e-02,
         -5.4932e-03, -1.2390e-02,  1.4771e-02,  9.5215e-03,  1.9531e-03,
          5.2185e-03, -1.9775e-02,  7.6294e-03,  8.0566e-03,  3.9673e-04,
         -3.8757e-03, -2.2705e-02,  6.1340e-03,  8.5449e-03,  2.5757e-02,
          7.2632e-03,  1.6479e-03,  1.3062e-02, -3.1738e-03,  3.0823e-03,
         -7.3853e-03,  1.2329e-02,  1.5137e-02,  9.2163e-03, -6.1035e-03,
          1.0925e-02,  4.3945e-03, -2.2583e-03, -3.0518e-04,  1.4404e-02,
          1.4343e-02,  1.5259e-03,  2.4536e-02, -1.4832e-02, -7.4768e-03,
         -2.5940e-03, -9.5215e-03, -5.7373e-03,  1.0498e-02, -2.1362e-03,
          7.4768e-03, -1.9836e-03,  1.5259e-02,  1.2207e-04, -1.0376e-02,
          3.6011e-03, -1.0986e-02,  4.9744e-03,  1.5320e-02, -3.8910e-03,
          5.4016e-03, -4.8218e-03,  2.9907e-02, -3.9673e-03, -4.6997e-03,
          5.8594e-03,  1.0010e-02, -6.8054e-03,  3.1586e-03, -1.8311e-02,
          2.7466e-03, -4.8523e-03,  3.9062e-03, -1.2329e-02, -3.4180e-03,
          4.6387e-03, -2.5024e-03, -1.8311e-03, -1.6846e-02, -6.4087e-03,
          2.4414e-02, -1.1902e-03,  7.5989e-03, -1.8311e-03, -9.8877e-03,
         -6.3171e-03, -2.5330e-03,  1.3550e-02, -9.2773e-03,  9.1553e-04],
        [ 9.6875e-01, -4.4062e+00, -9.2188e-01,  6.6250e+00,  4.0625e+00,
          1.4609e+00, -4.8125e+00,  3.5625e+00, -1.8047e+00, -4.7812e+00,
         -1.5869e-03, -1.0864e-02,  1.8311e-02, -7.2632e-03, -8.9111e-03,
         -1.0864e-02,  1.3428e-03, -2.6367e-02, -4.2419e-03, -2.6611e-02,
         -8.7891e-03,  1.1826e-03,  6.5308e-03,  1.0620e-02, -5.8899e-03,
         -8.6670e-03,  6.4697e-03,  6.4087e-03,  8.6060e-03, -1.3672e-02,
          3.7842e-03, -1.1230e-02,  2.7771e-03, -1.1841e-02, -7.6294e-04,
         -1.7090e-03,  1.1108e-02,  5.0964e-03,  6.1035e-05, -9.1553e-03,
          1.7944e-02, -5.0049e-03, -2.4780e-02,  1.9897e-02,  3.6621e-04,
          6.7749e-03,  2.7954e-02,  1.0376e-02, -1.7090e-03,  6.2256e-03,
         -2.0752e-02, -1.8799e-02,  1.5015e-02,  1.1536e-02, -2.3499e-03,
          1.4191e-03, -2.1240e-02,  1.2146e-02, -2.2339e-02,  2.9297e-03,
         -1.2512e-03, -5.8594e-03,  2.0020e-02, -2.3560e-02,  2.4414e-03,
          1.1597e-03, -3.8757e-03, -2.0142e-03,  2.2705e-02, -1.0986e-02,
         -5.2185e-03, -4.3335e-03,  1.4709e-02, -2.0996e-02, -6.5918e-03,
          4.8218e-03, -3.6926e-03, -5.8289e-03, -2.6398e-03, -1.3367e-02,
         -1.6602e-02,  4.8828e-04, -9.1553e-05, -9.2163e-03,  2.4414e-03,
          1.4404e-02, -1.3611e-02,  4.4556e-03,  8.6670e-03, -1.2329e-02,
          1.0681e-02,  3.5553e-03,  5.6763e-03,  2.5146e-02, -1.7822e-02,
         -1.0986e-02, -1.5259e-02, -1.3550e-02,  2.1973e-03, -1.6602e-02],
        [ 2.7031e+00,  6.3281e-01, -1.6484e+00, -3.0312e+00,  3.1875e+00,
         -4.0312e+00, -3.4531e+00, -1.6719e+00,  1.1562e+01, -4.1875e+00,
         -1.5076e-02, -6.8054e-03,  3.4790e-03, -1.1841e-02, -1.3428e-03,
         -2.5024e-02,  5.7983e-03,  7.1411e-03, -6.0120e-03, -2.6367e-02,
         -1.2817e-03,  2.0630e-02, -7.2021e-03,  2.7771e-03, -6.3171e-03,
          1.5259e-02,  0.0000e+00,  6.4087e-03, -9.3994e-03, -2.4109e-03,
         -1.1292e-02, -5.6152e-03,  5.1575e-03, -4.5166e-03,  1.0864e-02,
         -9.3994e-03, -2.7344e-02,  2.8076e-02, -1.3489e-02, -2.1484e-02,
          2.4170e-02, -1.3504e-03, -1.2817e-02,  3.8452e-03, -6.3171e-03,
          1.1353e-02, -1.4648e-02,  6.2256e-03, -3.1433e-03, -1.8921e-03,
         -1.1475e-02,  1.0071e-02,  1.0986e-03,  1.6602e-02, -1.3245e-02,
          9.2773e-03, -3.0273e-02,  1.5869e-02, -2.8687e-03, -1.9653e-02,
         -1.2573e-02,  1.3977e-02,  4.3030e-03, -3.4790e-03,  2.0996e-02,
          1.7700e-02,  9.3994e-03, -1.1719e-02,  4.1992e-02, -2.2827e-02,
         -2.0874e-02, -3.5400e-03, -2.0508e-02, -7.0190e-03,  1.4648e-03,
          2.1362e-02, -9.1553e-03, -8.5449e-03,  1.6846e-02, -1.3367e-02,
         -1.4771e-02, -1.0742e-02, -3.9673e-03, -6.1035e-03,  1.2207e-02,
          2.1729e-02, -1.5869e-02, -4.8523e-03,  6.0730e-03, -6.9580e-03,
          9.2773e-03, -8.0109e-04,  1.0498e-02,  1.2939e-02,  4.1504e-03,
         -1.0681e-03, -2.5757e-02,  1.5747e-02,  2.2217e-02, -1.4221e-02],
        [ 1.7188e+00, -3.0625e+00, -6.7812e+00, -8.5625e+00, -2.5312e+00,
         -6.5625e+00, -6.8438e+00, -1.7109e+00, -2.7344e+00,  3.7250e+01,
         -3.3203e-02,  2.6123e-02,  6.6528e-03, -2.3682e-02,  3.0518e-04,
         -2.2827e-02,  4.2725e-04,  4.0283e-03,  4.5776e-04, -1.5869e-02,
          4.0283e-03,  3.5706e-03,  9.3384e-03, -8.6670e-03,  8.4229e-03,
          1.5564e-03, -1.6479e-03, -1.6113e-02, -6.2256e-03, -7.8125e-03,
         -7.3242e-04, -1.8921e-02,  9.4604e-04,  6.1035e-05, -1.9043e-02,
         -1.9043e-02, -2.3193e-02,  8.6670e-03, -3.8574e-02, -3.1738e-02,
         -1.7090e-02,  7.2021e-03, -2.8381e-03, -2.4414e-02, -4.2725e-02,
         -5.9204e-03,  1.2817e-03, -3.0884e-02,  2.9907e-03, -6.8359e-03,
          3.1738e-02,  1.8311e-02, -8.5449e-03, -2.1667e-03, -1.4160e-02,
         -1.7090e-02, -4.7607e-03,  2.0996e-02,  2.9541e-02,  4.0283e-02,
          2.8320e-02, -1.5793e-03,  1.5137e-02,  1.8555e-02, -2.3071e-02,
          3.0396e-02, -1.2451e-02,  1.2573e-02,  1.5747e-02, -1.0254e-02,
          1.2634e-02, -3.4668e-02,  1.0742e-02, -2.8809e-02, -2.1240e-02,
         -7.6904e-03,  2.8076e-03, -9.8877e-03,  2.5391e-02,  1.5411e-03,
          1.6968e-02,  2.0752e-02,  7.3547e-03, -1.3794e-02,  3.7842e-02,
          5.7617e-02, -8.3618e-03,  1.9989e-03,  2.0630e-02,  1.4099e-02,
          1.9287e-02,  8.9722e-03, -1.4893e-02, -1.3062e-02, -1.5015e-02,
          6.5613e-03, -2.4292e-02,  1.3916e-02,  1.2817e-02, -4.5898e-02]])

def pairwise_dist(A, B):
    na = torch.sum(A**2, dim=1)
    nb = torch.sum(B**2, dim=1)

    na = na.reshape(-1, 1)
    nb = nb.reshape(1, -1)

    D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.T) + nb, torch.tensor(1e-12)))
    return D

def pairwise_cosine_similarity(A, B):
    # Get norms of the vectors
    A = A.to(torch.float16)
    B = B.to(torch.float16)
    A_norm = torch.linalg.norm(A, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=1, keepdim=True)

    # # Avoid division by zero
    # A_norm = torch.where(A_norm == 0, torch.tensor(1e-12, device=A.device), A_norm)
    # B_norm = torch.where(B_norm == 0, torch.tensor(1e-12, device=B.device), B_norm)

    A_normalized = A / A_norm
    B_normalized = B / B_norm

    # Calculate cosine similarity
    # print('pairwise shapes', A_normalized.shape, B_normalized.T.shape)
    # print('pairwise devices', A.device, B.device)
    similarity = torch.matmul(A_normalized, B_normalized.T)
    return 1-similarity

def normalize_tensor_vectors_vmap(tensor):
    return tensor / torch.linalg.norm(tensor, dim=1, keepdim=True)

def calculate_vector_norms(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return norms

def get_max_element(tensor):
    return torch.max(tensor)

def compute_centroids_new(z, in_y, num_classes=10):
    true_y = torch.argmax(in_y, dim=1)
    class_mask = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
    sum_z = torch.matmul(class_mask.T, z)
    count_per_class = class_mask.sum(dim=0)
    count_per_class = torch.clamp(count_per_class, min=1e-12)
    centroids = sum_z / count_per_class.unsqueeze(1)
    return centroids


def compute_centroids(z, in_y, num_classes=10):
    true_y = torch.argmax(in_y, dim=1)
    centroids = []
    
    for i in range(num_classes):
        class_i_mask = (true_y == i).float().unsqueeze(1)  # Create mask
        num_class_i = class_i_mask.sum()
        
        if num_class_i == 0:
            centroids.append(torch.zeros(z.shape[1], device=z.device))
        else:
            class_i_mask = torch.ones_like(z) * class_i_mask
            masked_z_i = z * class_i_mask
            centroid_i = masked_z_i.sum(dim=0) / num_class_i
            centroids.append(centroid_i)
    
    return torch.stack(centroids)


def update_learnt_centroids_new(learnt_y, centroids, decay_factor=1.0, norm_learnt_y: bool=False):
    nonzero_mask = torch.any(centroids != 0, dim=1)

    updated_centroids = torch.where(
        nonzero_mask.unsqueeze(1),  # Expand mask to match the second dimension
        centroids,
        learnt_y,
    )

    new_learnt_y = decay_factor * updated_centroids + (1 - decay_factor) * learnt_y
    if norm_learnt_y:
        new_learnt_y = normalize_tensor_vectors_vmap(new_learnt_y)

    return new_learnt_y

def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0, norm_learnt_y: bool=False, exp_centroid_decay_factor=0.0):
    num_classes, latent_dim = learnt_y.shape  # Get dimensions
    new_learnt_y = []
    
    for i in range(num_classes):
        enc_y = centroids[i]
        if torch.count_nonzero(enc_y) == 0:  # Check if all zero
            enc_y = learnt_y[i]
        new_enc_y = decay_factor * enc_y + (1 - decay_factor) * learnt_y[i]
        new_learnt_y.append(new_enc_y)

    if norm_learnt_y:
        new_learnt_y = normalize_tensor_vectors_vmap(new_learnt_y)
    
    return torch.stack(new_learnt_y)


def cos_repel_loss_z(z, in_y, num_labels):
    norm_z = z / torch.norm(z, dim=1, keepdim=True)
    cos_dist = torch.matmul(norm_z, norm_z.T)
    true_y = torch.argmax(in_y, dim=1).unsqueeze(1)

    # Create a mask for same-class pairs
    same_class_mask = torch.ones((in_y.shape[0], in_y.shape[0]), device=z.device, dtype=torch.float32)
    for i in range(num_labels):
        # Mask for class `i`
        true_y_i = (true_y == i).float()
        class_i_mask = 1 - torch.matmul(true_y_i, true_y_i.T)  # 0 if same class, 1 otherwise
        same_class_mask *= class_i_mask

    # Compute the loss: mean of cosine distances for different-class pairs
    # abs_cos_dist = torch.relu(cos_dist)
    return torch.mean(cos_dist * same_class_mask)


def cos_repel_loss_z_optimized(z, in_y):
    # Normalize the vectors
    norm_z = z / torch.norm(z, dim=1, keepdim=True)

    # Compute cosine similarity matrix
    cos_dist = torch.matmul(norm_z, norm_z.T)
    # adj_cos_dist = torch.relu(cos_dist)
    # p_dist = pairwise_dist(norm_z, norm_z)
    # learnt_y_dist = torch.relu(-torch.matmul(z, z))

    # Get the class labels (assumes one-hot encoded input)
    true_y = torch.argmax(in_y, dim=1)  # Shape: [batch_size]

    # Create a mask where same-class pairs are 0, and different-class pairs are 1
    true_y_expanded = true_y.unsqueeze(0)  # Shape: [1, batch_size]
    class_mask = (true_y_expanded != true_y_expanded.T).float()  # Shape: [batch_size, batch_size]

    # Compute the loss: mean of cosine distances for different-class pairs
    return torch.mean(cos_dist * class_mask)


def contrastive_loss(centroids):
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(centroids, centroids)
    # return loss.mean()
    detached_centroids = centroids.detach()
    contrastive_score = torch.einsum(
        "id, jd->ij",
        centroids, # / self.args.temperature,
        centroids,
        # detatched_centroids,
    )

    bsz = centroids.shape[0]
    labels = torch.arange(
        0, bsz, dtype=torch.long, device=contrastive_score.device
    )
    contrastive_loss = torch.nn.functional.cross_entropy(
        contrastive_score, labels
    )
    # print('contrastive_loss & score', contrastive_loss, contrastive_score)
    return contrastive_loss


def generate_random_orthogonal_vectors(num_classes, latent_dim, device, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    random_matrix = np.random.rand(latent_dim, num_classes)
    Q, _ = np.linalg.qr(random_matrix)
    orthogonal = torch.from_numpy(Q.T)
    orthogonal = orthogonal.to(device)
    return orthogonal


class LearningWithAdaptiveLabels(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self,
            latent_dim: int,
            num_classes: int,
            stationary_steps: int,
            device: torch.device,
            current_step: int = 1,
            decay_factor: float = 1.0,
            structure_loss_weight: float = 1.0,
            init_fn: str = 'onehot',
            pairwise_fn: str = 'dist',
            num_features: int = 2048,
            verbose: bool = False,
            early_stop: bool = Optional[int],
            lwal_centroid_freeze_steps: Optional[int] = None,
            exp_centroid_decay_factor: float = 0.0,
            exp_stationary_step_decay_factor: float = 0.0,
            # BCE args
            # smoothing=0.1,
            # target_threshold: Optional[float] = None,
            # weight: Optional[torch.Tensor] = None,
            # reduction: str = 'mean',
            # sum_classes: bool = False,
            # pos_weight: Optional[Union[torch.Tensor, float]] = None,
    ):
        super(LearningWithAdaptiveLabels, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.stationary_steps = stationary_steps
        self.current_step = current_step
        self.learnt_y = None
        match init_fn:
            case 'random':
                self.learnt_y = generate_random_orthogonal_vectors(num_classes, latent_dim, device) 
            case 'learnt':
                self.learnt_y = LAST_Z_OF_LABEL
            case _:
                self.learnt_y = torch.eye(num_classes, latent_dim, device=device)
        print(self.learnt_y)
        self.decay_factor = decay_factor
        self.structure_loss_weight = structure_loss_weight
        self.pairwise_fn_name = pairwise_fn
        self.pairwise_fn = pairwise_cosine_similarity if pairwise_fn == 'cos' else pairwise_dist
        self.verbose = verbose
        self.early_stop = early_stop
        self.lwal_centroid_freeze_steps = lwal_centroid_freeze_steps
        self.exp_centroid_decay_factor = exp_centroid_decay_factor
        self.exp_stationary_step_decay_factor = exp_stationary_step_decay_factor
        self.maximum_element = 0
        self.maximum_norm = 0
        self.last_z_of_label = torch.zeros(num_classes, latent_dim, device=device)
    
    def get_learnt_y(self):
        return self.learnt_y

    def cross_entropy_pull_loss(self, enc_x, in_y, learnt_y):
        enc_x_dist = self.pairwise_fn(enc_x, learnt_y)
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
        loss = torch.sum(-in_y * logits, dim=-1)
        return loss.mean()

    def cross_entropy_nn_pred(self, enc_x, in_y, learnt_y):
        """Cross Entropy NN Prediction based on learnt_y."""
        enc_x_dist = self.pairwise_fn(enc_x, learnt_y)
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
        preds = torch.argmax(logits, dim=1)
        true_y = torch.argmax(in_y, dim=1)
        return preds, true_y

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        z = x.clone()
        self.device = x.device
        num_labels = self.num_classes
        structure_loss = 0
        stationary_steps_adj = self.stationary_steps
        update_centroids = (self.current_step % int(self.stationary_steps) == 0)
        # For freezing experiment
        update_centroids = update_centroids and (self.lwal_centroid_freeze_steps is None or self.current_step <= self.lwal_centroid_freeze_steps)
        # For experiment C 
        # update_centroids = update_centroids and ((self.current_step // 195) > 19)
        if update_centroids:
            centroids = compute_centroids(x, target, self.num_classes)
            structure_loss = contrastive_loss(centroids)
            centroids = centroids.detach()
            adj_decay_factor = self.decay_factor * math.exp(self.current_step / self.stationary_steps * self.exp_centroid_decay_factor)
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids, adj_decay_factor, self.pairwise_fn == 'cos', self.exp_centroid_decay_factor)
            # print(self.learnt_y)
            # structure_loss = cos_repel_loss_z_optimized(x, target)
            # self.stationary_steps *= math.exp(self.exp_stationary_step_decay_factor)
            # self.decay_factor *= math.exp(self.exp_centroid_decay_factor)
            print('new stationary_steps and decay_factor', stationary_steps, decay_factor)

        if self.early_stop and self.current_step == (self.early_stop*195):
            if self.verbose: 
                print('learnt_y (near the end of training)')
                print(self.learnt_y)
            print('pairwise cosine sim of learnt_y x learnt_y')
            print(pairwise_cosine_similarity(self.learnt_y, self.learnt_y))
            print("last z's of each label (to be used as centroids for next run)")
            print(self.last_z_of_label)
            raise KeyboardInterrupt()
        
        self.maximum_element = max(self.maximum_element, get_max_element(z))
        self.maximum_norm = max(self.maximum_norm, get_max_element(calculate_vector_norms(z)))
        # Accuracy prints (every 50 steps)
        if (self.current_step % 5) == 1 and self.verbose: 
            print('train_acc @ %s steps' % self.current_step, self.acc_helper(z, target, self.learnt_y))
        # Experiment C
        if (self.current_step // 195) == 19:
            # print(target)
            idx = torch.argmax(target, dim=-1)
            # self.last_z_of_label[idx] = z.detach()
            for i in range(z.size(0)):
                label = idx[i].item()
                self.last_z_of_label[label] = z[i].detach()
        self.current_step += 1
        # Experiment C
        # if self.current_step == 3901:
        #     print("Switching over centroids mode")
        #     self.learnt_y = self.last_z_of_label
        #     print("Centroids are: ", self.learnt_y)
        # # Print data every epoch.
        # if (self.current_step % 195) == 194 and self.verbose:
        #     print('z', self.maximum_element, self.maximum_norm, z)
        #     if self.pairwise_fn == pairwise_cosine_similarity:
        #         cossim = pairwise_cosine_similarity(normalize_tensor_vectors_vmap(z), self.learnt_y)
        #         print('cosine sim', 
        #             get_max_element(-cossim),
        #             get_max_element(calculate_vector_norms(-cossim)),
        #             cossim)
        #     else:
        #         dists = pairwise_dist(z, self.learnt_y)
        #         normed_dists = pairwise_dist(normalize_tensor_vectors_vmap(z), normalize_tensor_vectors_vmap(self.learnt_y))
        #         print('dists', 
        #               get_max_element(dists),
        #               get_max_element(calculate_vector_norms(dists)),
        #               dists)
        #         print('normed_dists', 
        #               get_max_element(normed_dists),
        #               get_max_element(calculate_vector_norms(normed_dists)),
        #               normed_dists)
        input_loss = self.cross_entropy_pull_loss(x, target, self.learnt_y)
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss, self.learnt_y
    
    def test(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        z = x.clone()
        self.device = x.device

        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        # one_hot_target.to
        input_loss = self.cross_entropy_pull_loss(z, one_hot_target, self.learnt_y)
        # structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        structure_loss = 0
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss

    def acc_helper(self, z, target, learnt_y):
        pred_y, true_y = self.cross_entropy_nn_pred(z, target, learnt_y)
        acc1 = (pred_y == true_y).float().mean() * 100.
        return acc1

    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        z = output.clone()
        z = z.to(torch.float32)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        acc1 = self.acc_helper(z, one_hot_target, learnt_y)
        return acc1, structure_loss
