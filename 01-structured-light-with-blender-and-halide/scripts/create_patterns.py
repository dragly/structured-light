from pylab import zeros, ones, indices, uint8
from PIL import Image
rows = 1024
patterns = 7
data = zeros((patterns, rows, rows))
for i in range(patterns):
    divisions = 2 ** (i + 1)
    step = rows / divisions
    period = 2 * step
    data[i] = ones((rows, rows)) * ((indices((rows, rows))[1] % period) > step)
    img = Image.fromarray(255 * data[i].astype(uint8), mode='L')
    img.save("output/pattern_{}.png".format(i + 1))
