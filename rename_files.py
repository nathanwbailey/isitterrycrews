import os
from lxml import etree

# for idx, f in enumerate(os.listdir('images/terry')):
#     print(f)
#     print(idx)
#     os.rename('images/terry/'+f, 'images/terry/'+'terry_'+str(idx)+'.jpg')

# for idx, f in enumerate(os.listdir('images/non_terry')):
#     os.rename('images/non_terry/'+f, 'images/non_terry/'+'non_terry_'+str(idx)+'.jpg')


num_images_terry = len(os.listdir(path='images/terry'))
num_images_non_terry = len(os.listdir(path='images/non_terry'))

root = etree.Element("Images")
items = etree.SubElement(root, "Items", num_images=str(num_images_terry+num_images_non_terry))
list_files = os.listdir(path='images/terry')
list_files.sort()
for f in list_files:
    id = f
    label = '0'
    item = etree.SubElement(items, "Item", imageName=id, label=label)

list_files = os.listdir(path='images/non_terry')
list_files.sort()
for f in list_files:
    id = f
    label = '1'
    item = etree.SubElement(items, "Item", imageName=id, label=label)


tree = etree.ElementTree(root)
tree.write("images/image_labels.xml", pretty_print=True)
