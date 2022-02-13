import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET


data_path = ''

parser = argparse.ArgumentParser(description='rescale_prop')
parser.add_argument('-dir', dest="data_dir", required=False,  default='../data/dataset_jpg')
parser.add_argument('-size', dest='size', type=int, required=False,  default='320')
args = parser.parse_args()

print(args.data_dir, args.size)

IMAGE_SIZE = (args.size, args.size)

dest_path = '../data/dataset_prop'

add_path = ''

def parse_xml(filename, r):

  max_inst = 0
  #print(filename)
  tree = ET.parse(filename)
  root = tree.getroot()

  root.find('filename').text = root.find('filename').text[:-3] + 'jpg'
  root.find('size').find('width').text = str(IMAGE_SIZE[0])
  root.find('size').find('height').text = str(IMAGE_SIZE[1])

  for object in root.findall('object'):
    object.find('bndbox').find('xmin').text = str(int(int(object.find('bndbox').find('xmin').text) * r))
    object.find('bndbox').find('ymin').text = str(int(int(object.find('bndbox').find('ymin').text) * r))
    object.find('bndbox').find('xmax').text = str(int(int(object.find('bndbox').find('xmax').text) * r))
    object.find('bndbox').find('ymax').text = str(int(int(object.find('bndbox').find('ymax').text) * r))
  #print(len(root.findall('object')))
  if len(root.findall('object')) > 99:
    print(filename)

  #   item.text = name
  #   print(item.text)
  # for item in root.findall('filename'):
  #   name = item.text[:-4] + '.jpg'
  #   item.text = name
  #print(root.find('filename').text)
  save_filename = filename[:16] + 'prop' + filename[19:]
  #print(save_filename)
  #tree.write(save_filename)




def resize_proportionally(image):
  #cv2_imshow(image)

  (h, w) = image.shape[:2]
  #print('orig ', image.shape[:2])
  rescaled_width = w
  rescaled_height = h
  resized_image = image
  if h > IMAGE_SIZE[0] or w > IMAGE_SIZE[0] or h < IMAGE_SIZE[0] or w < IMAGE_SIZE[0]:
    if(h > w):
      r = IMAGE_SIZE[0]/h
    if(w > h):
      r = IMAGE_SIZE[0]/w
    if(h==w):
      r = IMAGE_SIZE[0]/w
    rescaled_height = r * h
    rescaled_width = w * r
    rescaled_width = int(rescaled_width)
    rescaled_height = int(rescaled_height)
    resized_image = cv2.resize(image, (rescaled_width, rescaled_height))
    #print('resized ', resized_image.shape[:2])

  blank_image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0], 3), np.uint8)
  blank_image[:rescaled_height, :rescaled_width, :3] = resized_image
  return blank_image, r


add_path = ''
def dirs(path, add_path):
  for item in os.listdir(path):
    if item == 'annotations':
      continue
    dir_path = path +'/' + item
    #os.mkdir(dest_path + item)
    #print(dest_path+item)

    if os.path.isdir(dir_path):
      # рекурсивно заходить в директории

      #print(item)
      if item == 'test':
        add_path = ''
      add_path += '/' + item
      #print(add_path)
      dirs(dir_path, add_path)
    else:

      item_path = path + '/' + item

      img = cv2.imread(item_path)
      resized_img, r = resize_proportionally(img)


      xml_path = path[:-6] + 'annotations' + '/' + item[:-4] + '.xml'
      parse_xml(xml_path, r)

      save_path = dest_path + add_path + '/' + item
      #cv2.imwrite(save_path, resized_img)

dirs(args.data_dir, add_path)






  #print(blank_image.shape[:2])
  #cv2_imshow(blank_image)
  #rescaled_write_path = rescale_path + '/pers_'+ str(i) + '.jpg'
  #print(rescaled_write_path)
  #cv2.imwrite(rescaled_write_path, blank_image)