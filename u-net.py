import torch 
from torch import nn

print(torch.__version__)

def center_crop(enc_feat, target_tensor):
    _, _, h, w = target_tensor.shape
    _, _, H, W = enc_feat.shape

    start_h = (H - h) // 2
    start_w = (W - w) // 2

    return enc_feat[:, :, start_h:start_h + h, start_w:start_w + w]

class Unet(nn.Module):
    
    def __init__(self):
        super().__init__()

        # encoder below
        self.conv_layer_64_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv_layer_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)


        # size formula  Output size= [(N−F+2P) / S] + 1 
        # e.g [(572 - 3 + 2*0) / 1] + 1 = 570
        # e.g [(570 - 3 + 2*0) / 1] + 1 = 568
                                                                   
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_layer_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # max pooling

        self.conv_layer_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv_layer_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        # max pooling

        self.conv_layer_512_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv_layer_512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # max pooling

        self.conv_layer_1024_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv_layer_1024_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)


        # decoder below

        # When you apply a ConvTranspose2d (i.e., up-convolution), the output size is determined by this formula:
        # Output Size= (I−1) × S − 2P + K + output_padding
        # Hout =[(Hin−1)×stride]−[2×padding]+kernel_size+output_padding
        # Hout = [27*2] - [2*0] + 2 + 0 = 56

        # I: Input size (height or width)
        # S: Stride
        # P: Padding
        # K: Kernel size
        # output_padding: Extra padding added to match a specific size (usually 0 or 1)

        # UpConv 2x2 (transposed conv)
        self.upconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        
        self.conv_512_1 = nn.Conv2d(in_channels=512 + 512, out_channels=512, kernel_size=3)
        self.conv_512_2 = nn.Conv2d(in_channels=512 , out_channels=256, kernel_size=3)  


        self.upconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

        self.conv_256_1 = nn.Conv2d(in_channels=256 + 256, out_channels=256, kernel_size=3)
        self.conv_256_2 = nn.Conv2d(in_channels=256 , out_channels=128, kernel_size=3)


        self.upconv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        self.conv_128_1 = nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=3)
        self.conv_128_2 = nn.Conv2d(in_channels=128 , out_channels=64, kernel_size=3)


        self.upconv_4 = nn.ConvTranspose2d(in_channels=64 , out_channels=64, kernel_size=2, stride=2)

        self.conv_64_1 = nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=3)
        self.conv_64_2 = nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=3)



        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)





    
    def forward(self, image):

        print(image.size())
        # decoder 
        x1 = self.conv_layer_64_1(image)
        print(x1.size())
        x2 = self.conv_layer_64_2(x1)
        print(x2.size())
        x3 = self.max_pool_2x2(x2)
        print(x3.size())

        x4 = self.conv_layer_128_1(x3)
        print(x4.size())
        x5 = self.conv_layer_128_2(x4)
        print(x5.size())
        x6 = self.max_pool_2x2(x5)
        print(x6.size())

        x7 = self.conv_layer_256_1(x6)
        print(x7.size())
        x8 = self.conv_layer_256_2(x7)
        print(x8.size())
        x9 = self.max_pool_2x2(x8)
        print(x9.size())

        print("")    
        x10 = self.conv_layer_512_1(x9)
        print(x10.size())
        x11 = self.conv_layer_512_2(x10)
        print(x11.size())
        x12 = self.max_pool_2x2(x11)
        print(x12.size())

        print("")    
        x13 = self.conv_layer_1024_1(x12)
        print(x13.size())
        x14 = self.conv_layer_1024_2(x13)
        print(x14.size())

        #decoder
        print("")
        y1 = self.upconv_1(x14)
        print(y1.size())

        x11_cropped = center_crop(x11, y1)
        y1_cat = torch.cat([y1,x11_cropped], dim=1)
        print(y1_cat.size())

        y2 = self.conv_512_1(y1_cat)
        print(y2.size())
        y3 = self.conv_512_2(y2)
        print(y3.size())

        print("")
        y4 = self.upconv_2(y3)
        print(y4.size())

        x8_cropped = center_crop(x8, y4)
        y5_cat = torch.cat([y4,x8_cropped], dim=1)
        print(y5_cat.size())

        print("")
        y6 = self.conv_256_1(y5_cat)
        print(y6.size())
        y7 = self.conv_256_2(y6)
        print(y7.size())

        print("")
        y8 = self.upconv_3(y7)
        print(y8.size())

        x5_cropped = center_crop(x5, y8)
        y9_cat = torch.cat([y8,x5_cropped], dim=1)
        print(y9_cat.size())

        print("")
        y10 = self.conv_128_1(y9_cat)
        print(y10.size())
        y11 = self.conv_128_2(y10)
        print(y11.size())




        print("")
        y12 = self.upconv_4(y11)
        print(y12.size())

        x2_cropped = center_crop(x2, y12)
        y13_cat = torch.cat([y12,x2_cropped], dim=1)
        print(y13_cat.size())

        print("")
        y14 = self.conv_64_1(y13_cat)
        print(y14.size())
        y15 = self.conv_64_2(y14)
        print(y15.size())


        out = self.final_conv(y15)
        output = torch.sigmoid(out)

        return output 
        



image = torch.rand((1,1,572, 572))
model = Unet()
print(model(image))


