from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU, StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

import numpy as np
from .pytorch_revgrad import RevGrad

class My_UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

        # we store some things that a potential domain classifier needs
        self.conv_op = encoder.conv_op
        self.norm_op = encoder.norm_op
        self.norm_op_kwargs = encoder.norm_op_kwargs
        self.nonlin = encoder.nonlin
        self.nonlin_kwargs = encoder.nonlin_kwargs
        self.dropout_op = encoder.dropout_op
        self.dropout_op_kwargs = encoder.dropout_op_kwargs
        self.conv_bias = encoder.conv_bias
        self.kernel_sizes = encoder.kernel_sizes

        self.output_channels = encoder.output_channels

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        stage_outputs = [] # JR
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            stage_outputs.append(x) # JR
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r, stage_outputs

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class DomainClassifier(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder],
                 alpha: float,
                 ):
        super().__init__()
        self.encoder = encoder
        self.alpha = alpha

        # JR: the last layer output of encoder as feature
        input_feature = encoder.output_channels[-1]
        num_convs = 2 # number of conv per block
        output_channels = 100 # conv kernel channel
        domain_channels = 2 # output domain label channel

        self.grl = RevGrad(alpha = self.alpha)
        self.doubleConvs = nn.Sequential(
            ConvDropoutNormReLU(
                encoder.conv_op, input_feature, output_channels, (3, 3, 3), 1, encoder.conv_bias, encoder.norm_op,
                encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                encoder.nonlin_kwargs, False
            ),
            *[
                ConvDropoutNormReLU(
                    encoder.conv_op, output_channels, output_channels, (3, 3, 3), 1, encoder.conv_bias, encoder.norm_op,
                    encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                    encoder.nonlin_kwargs, False
                )
                for i in range(1, num_convs)
            ]
        )

        self.doubleConvs2 = nn.Sequential(
            *[
                ConvDropoutNormReLU(
                    encoder.conv_op, output_channels, output_channels, (3, 3, 3), 1, encoder.conv_bias, encoder.norm_op,
                    encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                    encoder.nonlin_kwargs, False
                )
                for i in range(0, num_convs)
            ]
        )

        self.MaxPool3d = nn.MaxPool3d(2) # downsample by 2
        self.Flatten = nn.Flatten()
        self.Dense1 = nn.Sequential(
                        nn.LazyLinear(out_features=output_channels),
                        nn.ReLU()
                    )
        self.Dense2 = nn.Linear(in_features=output_channels, out_features=domain_channels)

    def forward(self, skips):
        # alpha is here since it might change during the training
        lres_input = skips[-1]
        x = self.grl(lres_input)
        x = self.doubleConvs(x)
        x = self.MaxPool3d(x)
        x = self.doubleConvs2(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        return self.Dense2(x)

    def set_alpha(self, new_alpha: float):
        self.alpha = new_alpha
        self.grl.set_alpha(self.alpha)

    def get_alpha(self):
        return self.grl.get_alpha()

class PlainConvUNet_DA(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 alpha: float,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.alpha = alpha
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        # JR: establish domain classifier, need encoder to infer last layer feature
        self.domain_classifier = DomainClassifier(self.encoder, alpha=self.alpha)

    # JR: for each epoch/step, alpha might change
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips), self.domain_classifier(skips)

class DomainClassifier_onDecoder(nn.Module):
    def __init__(self,
                 decoder: Union[My_UNetDecoder],
                 alpha: float,
                 on_ith_decoder: int,
                 num_convblock: int,
                 ):
        super().__init__()
        self.decoder = decoder
        self.alpha = alpha
        self.on_ith_decoder = on_ith_decoder
        self.num_convblock = num_convblock

        input_feature = decoder.output_channels[-(on_ith_decoder + 1)] # number of input is based on stage
        num_convs = 2 # number of conv per block
        output_channels = 100 # conv kernel channel
        domain_channels = 2 # output domain label channel

        self.grl = RevGrad(alpha = self.alpha)
        self.doubleConvsMaxPool = nn.Sequential(
            ConvDropoutNormReLU(
                decoder.conv_op, input_feature, output_channels, (3, 3, 3), 1, decoder.conv_bias, decoder.norm_op,
                decoder.norm_op_kwargs, decoder.dropout_op, decoder.dropout_op_kwargs, decoder.nonlin,
                decoder.nonlin_kwargs, False
            ),
            *[
                ConvDropoutNormReLU(
                    decoder.conv_op, output_channels, output_channels, (3, 3, 3), 1, decoder.conv_bias, decoder.norm_op,
                    decoder.norm_op_kwargs, decoder.dropout_op, decoder.dropout_op_kwargs, decoder.nonlin,
                    decoder.nonlin_kwargs, False
                )
                for i in range(1, num_convs)
            ],
            nn.MaxPool3d(2),
        )

        self.doubleConvs2 = nn.Sequential(
            *[
                ConvDropoutNormReLU(
                    decoder.conv_op, output_channels, output_channels, (3, 3, 3), 1, decoder.conv_bias, decoder.norm_op,
                    decoder.norm_op_kwargs, decoder.dropout_op, decoder.dropout_op_kwargs, decoder.nonlin,
                    decoder.nonlin_kwargs, False
                )
                for i in range(0, num_convs)
            ]
        )

        self.doubleConvsMaxPool2 = nn.Sequential(
            *[
                ConvDropoutNormReLU(
                    decoder.conv_op, output_channels, output_channels, (3, 3, 3), 1, decoder.conv_bias, decoder.norm_op,
                    decoder.norm_op_kwargs, decoder.dropout_op, decoder.dropout_op_kwargs, decoder.nonlin,
                    decoder.nonlin_kwargs, False
                )
                for i in range(0, num_convs)
            ],
            nn.MaxPool3d(2),
        )

        # self.MaxPool3d = nn.MaxPool3d(2) # downsample by 2
        self.Flatten = nn.Flatten()
        self.Dense1 = nn.Sequential(
                        nn.LazyLinear(out_features=output_channels),
                        nn.ReLU()
                    )
        self.Dense2 = nn.Linear(in_features=output_channels, out_features=domain_channels)

    def forward(self, stage_outputs):
        lres_input = stage_outputs[self.on_ith_decoder - 1]
        x = self.grl(lres_input)
        x = self.doubleConvsMaxPool(x)
        for stage in range(self.num_convblock - 2): # if num_convblock >2, add more doubleConvsMaxPool layer
            x = self.doubleConvsMaxPool2(x)
        x = self.doubleConvs2(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        return self.Dense2(x)

    def set_alpha(self, new_alpha: float):
        self.alpha = new_alpha
        self.grl.set_alpha(self.alpha)

    def get_alpha(self):
        return self.grl.get_alpha()

class PlainConvUNet_DAonDecoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 alpha: float, # param for GSL
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 on_ith_decoder: int = 4,  # domain classifier is taking ith stage of decoder output (from [1,2,3,4])
                 num_convblock_domain_classifier: int = 2,
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.alpha = alpha
        self.on_ith_decoder = on_ith_decoder
        self.num_convblock = num_convblock_domain_classifier
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = My_UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                      nonlin_first=nonlin_first)

        # JR: establish domain classifier on decoder output: 1~4 for a 5 stages decoder, not for last stage, from neckbottle
        self.domain_classifier = DomainClassifier_onDecoder(self.decoder, alpha = self.alpha, on_ith_decoder = self.on_ith_decoder, num_convblock = self.num_convblock)

    # JR: for each epoch/step, alpha might change
    def forward(self, x):
        skips = self.encoder(x)
        outputs, stage_outputs = self.decoder(skips)

        return outputs, self.domain_classifier(stage_outputs)

