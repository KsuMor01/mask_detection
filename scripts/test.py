from PIL import Image, ImageShow, ImageDraw

img = Image.open('/home/ksumor/mask_detection/data/dataset/test/images/maksssksksss0.png')

draw = ImageDraw.Draw(img)

draw.rectangle([0,0,100,100], outline="yellow", width=3)
ImageShow.show(img)

