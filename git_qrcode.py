import qrcode

img = qrcode.make("https://github.com/vdmlovmeomeo/editeh")
img.save("my_qrcode.png")
