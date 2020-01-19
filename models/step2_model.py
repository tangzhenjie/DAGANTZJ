import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss

class Step2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=19, help='for determining the class number')
        if is_train:
            parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='the type of GAN objective.')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]    # "loss_"
        self.visual_names = ["imageB", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imageA", "fakeB", "imageB_idt",
                                  "pixelfakeB_out", "pixelimageB_idt_out"]  # ""

        self.model_names = ['generator']  # "net"
        if self.isTrain:
            self.model_names += ['pixel_discriminator', 'fc_discriminator']  # "net"

        # 特征生成器
        self.netgenerator = networks.generator(num_cls=opt.num_classes, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # 像素空间判别器
            self.netpixel_discriminator = networks.define_D(3, 64, 'basic', gpu_ids=self.gpu_ids)

            # 输出空间判别器
            self.netfc_discriminator = networks.fc_discriminator(num_classes=opt.num_classes, gpu_ids=self.gpu_ids)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

            # 像素判别器损失
            self.mse_loss = networks.GANLoss("lsgan").to(self.device)

            # 输出空间判别器损失
            self.bce_loss = networks.GANLoss("vanilla").to(self.device)

            # idt 损失
            self.generator_criterion = networks.GeneratorLoss().to(self.device)

            # 内容一致损失
            self.L1_loss = nn.L1Loss().to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netgenerator.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netpixel_discriminator.parameters(),
                                                                self.netfc_discriminator.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            self.imageA = input["A_image"].to(self.device)
            self.label = input["A_label"].to(self.device)
            self.imageB = input["B_image"].to(self.device)
        else:
            self.imageB = input["B_image"].to(self.device)
            self.label = input["B_label"].to(self.device)

    def forward(self):
        if self.isTrain:
            # iamgeA 通过 generator
            self.feature_A, _, self.fakeB = self.netgenerator(self.imageA)
            self.fakeB_cut = self.fakeB.detach() # 隔断反向传播
            self.feature_fakeB_cut, self.pre, _ = self.netgenerator(self.fakeB_cut) # 还没有使用
            self.pre_A_cut = self.pre.detach()  # 隔断反向传播
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

            #imageB 通过 generator
            _, self.pre_B, self.imageB_idt = self.netgenerator(self.imageB)
            self.imageB_idt_cut = self.imageB_idt.detach() # 隔断反向传播
            self.pre_B_cut = self.pre_B.detach()  # 隔断反向传播

            # imagesrA 通过判别器
            self.pixelfakeB_out = (F.tanh(self.netpixel_discriminator(self.fakeB)) + 1) * 0.5
            self.pixelimageB_idt_out = (F.tanh(self.netpixel_discriminator(self.imageB_idt)) + 1) * 0.5

            self.fcreal_out = self.netfc_discriminator(F.softmax(self.pre_B, dim=1))
        else:
            _, self.pre, _  = self.netgenerator(self.imageB)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
    def backward(self):
        """计算两个损失"""

        # 分割损失
        self.loss_cross_entropy = self.loss_function(self.pre, self.label.long().squeeze(1))

        # 像素级对齐损失
        self.loss_da1 = self.mse_loss(self.pixelfakeB_out, True)

        # 像素级对齐损失
        self.loss_da2 = self.mse_loss(self.pixelimageB_idt_out, True)

        # 输出空间对齐
        self.loss_da3 = self.bce_loss(self.fcreal_out, False)


        # 超分辨损失
        self.loss_idtB = self.L1_loss(self.imageB_idt, self.imageB)

        # A内容一致性损失
        self.loss_idtA = self.generator_criterion(self.fakeB, self.imageA, is_sr=False)

        # fix_pointA loss
        self.loss_fix_point = self.L1_loss(self.feature_A, self.feature_fakeB_cut)

        loss_DA = self.loss_da1 + self.loss_da2 + self.loss_da3
        loss_ID = self.loss_idtB + self.loss_idtA

        # 求分割损失和超分辨损失的和
        self.loss_G = loss_DA  * 1 + loss_ID * 10 + self.loss_cross_entropy * 10 +self.loss_fix_point * 5
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):

        pixeltrueB_out = (F.tanh(self.netpixel_discriminator(self.imageB)) + 1) * 0.5

        self.loss_D_da1 = self.mse_loss((F.tanh(self.netpixel_discriminator(self.fakeB_cut)) + 1) * 0.5, False) \
                          + self.mse_loss(pixeltrueB_out, True)
        self.loss_D_da2 = self.mse_loss((F.tanh(self.netpixel_discriminator(self.imageB_idt_cut)) + 1) * 0.5, False) \
                          + self.mse_loss(pixeltrueB_out, True)
        self.loss_D_da3 = self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_A_cut, dim=1)), False) \
                          + self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_B_cut, dim=1)), True)
        self.loss_D = (self.loss_D_da1 + self.loss_D_da2 + self.loss_D_da3) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

