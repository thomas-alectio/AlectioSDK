import os
import pandas as pd

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


os.system("wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
os.system("wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
os.system("wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
os.system("wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

os.system("gunzip t10k-images-idx3-ubyte.gz")
os.system("gunzip train-labels-idx1-ubyte.gz")
os.system("gunzip train-images-idx3-ubyte.gz")
os.system("gunzip t10k-labels-idx1-ubyte.gz")

print('Converting mnist images to .csv, this may take a minute...')

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "./mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "./mnist_test.csv", 10000)

df_orig_train = pd.read_csv('mnist_train.csv')
df_orig_test = pd.read_csv('mnist_test.csv')

df_orig_train.rename(columns={'5': 'label'}, inplace=True)
df_orig_test.rename(columns={'7': 'label'}, inplace=True)

df_orig_train.to_csv('mnist_train.csv', index=False)
df_orig_test.to_csv('mnist_test.csv', index=False)

print('Conversion complete')

os.system("mv mnist_train.csv ./data")
os.system("mv mnist_test.csv ./data")
os.system("mv train-images-idx3-ubyte ./data")
os.system("mv train-labels-idx1-ubyte ./data")
os.system("mv t10k-images-idx3-ubyte ./data")
os.system("mv t10k-labels-idx1-ubyte ./data")
