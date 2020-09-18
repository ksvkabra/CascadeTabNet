from PIL import Image


data = [[53.374203,   69.40673, 326.84393,   448.1108,      0.9999999]]
data1 = [[52,   68, 280,   385]]

print(data)

im = Image.open(
    r'/home/keshav/workspace/wisflux/table-detect/CascadeTabNet/Demo/file.png')

left = data[0][0]
top = data[0][1]
right = data[0][2]
bottom = data[0][3]

im1 = im.crop((left, top, right, bottom))

im1.show()

left = data1[0][0]
top = data1[0][1]
right = data1[0][2]
bottom = data1[0][3]

im1 = im.crop((left, top, right, bottom))

im1.show()
