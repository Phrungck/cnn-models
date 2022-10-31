import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # for first convolution module
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 3, stride=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())
        # finception module 1 (32+32 filters)
        self.conv2 = nn.Sequential(nn.Conv2d(96, 32, 1, stride=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(96, 32, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        # inception module 2 (32+48 filters)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 32, 1, stride=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 48, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU())
        # downsample module 1(80 filters)
        self.conv6 = nn.Sequential(nn.Conv2d(80, 80, 3, stride=2),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU())
        self.max1 = nn.MaxPool2d(3, 2)  # can be applied to all downsamples
        # inception module 3 (112+48 filters)
        self.conv7 = nn.Sequential(nn.Conv2d(160, 112, 1, stride=1),
                                   nn.BatchNorm2d(112),
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(160, 48, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU())
        # inception module 4 (96+64 filters)
        self.conv9 = nn.Sequential(nn.Conv2d(160, 96, 1, stride=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(160, 64, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        # inception module 5 (80+80 filters)
        self.conv11 = nn.Sequential(nn.Conv2d(160, 80, 1, stride=1),
                                    nn.BatchNorm2d(80),
                                    nn.ReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(160, 80, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(80),
                                    nn.ReLU())
        # inception module 6 (48+96 filters)
        self.conv13 = nn.Sequential(nn.Conv2d(160, 48, 1, stride=1),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU())
        self.conv14 = nn.Sequential(nn.Conv2d(160, 96, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU())
        # downsample module 2 (96 filters)
        self.conv15 = nn.Sequential(nn.Conv2d(144, 96, 3, stride=2),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU())
        # inception module 7 (176+160 filters)
        self.conv16 = nn.Sequential(nn.Conv2d(240, 176, 1, stride=1),
                                    nn.BatchNorm2d(176),
                                    nn.ReLU())
        self.conv17 = nn.Sequential(nn.Conv2d(240, 160, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(160),
                                    nn.ReLU())
        # inception module 8 (176+160 filters)
        self.conv18 = nn.Sequential(nn.Conv2d(336, 176, 1, stride=1),
                                    nn.BatchNorm2d(176),
                                    nn.ReLU())
        self.conv19 = nn.Sequential(nn.Conv2d(336, 160, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(160),
                                    nn.ReLU())
        # Mean pooling (7 kernels)
        self.avg = nn.AvgPool2d(7, padding=1)
        # Fully connected layers (10-way outputs)
        self.fc = nn.Linear(336, 10)

    def forward(self, x):
        x = self.conv1(x)
        merge_1 = self.conv2(x)
        merge_2 = self.conv3(x)
        x = torch.cat((merge_1, merge_2), dim=1)
        merge_3 = self.conv4(x)
        merge_4 = self.conv5(x)
        x = torch.cat((merge_3, merge_4), dim=1)
        merge_5 = self.conv6(x)
        merge_6 = self.max1(x)
        x = torch.cat((merge_5, merge_6), dim=1)
        merge_7 = self.conv7(x)
        merge_8 = self.conv8(x)
        x = torch.cat((merge_7, merge_8), dim=1)
        merge_9 = self.conv9(x)
        merge_10 = self.conv10(x)
        x = torch.cat((merge_9, merge_10), dim=1)
        merge_11 = self.conv11(x)
        merge_12 = self.conv12(x)
        x = torch.cat((merge_11, merge_12), dim=1)
        merge_13 = self.conv13(x)
        merge_14 = self.conv14(x)
        x = torch.cat((merge_13, merge_14), dim=1)
        merge_15 = self.conv15(x)
        merge_16 = self.max1(x)
        x = torch.cat((merge_15, merge_16), dim=1)
        merge_17 = self.conv16(x)
        merge_18 = self.conv17(x)
        x = torch.cat((merge_17, merge_18), dim=1)
        merge_19 = self.conv18(x)
        merge_20 = self.conv19(x)
        x = torch.cat((merge_19, merge_20), dim=1)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = GoogLeNet()
