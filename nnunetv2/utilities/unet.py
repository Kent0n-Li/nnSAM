from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.utilities.unet_decoder import UNetDecoder, SAMDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.paths import nnUNet_raw
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.modeling.tiny_vit_sam import TinyViT
import requests
import os

def download_model(url,destination):

    chunk_size = 8192  # Size of each chunk in bytes

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("Weights downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)




class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
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
                 nonlin_first: bool = False
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
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)




class SAMConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
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
                 nonlin_first: bool = False
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
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        

        save_path = nnUNet_raw
        model_weight_path = os.path.join(save_path, "mobile_sam.pt")

        if not os.path.exists(model_weight_path):
            download_model(url = 'https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt', destination= model_weight_path)


        model_type = "vit_t"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=model_weight_path)
        mobile_sam.to(device=device)
        
        self.sam_image_encoder = mobile_sam.image_encoder

        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False


    def forward(self, x):

        sam_input = x.detach()
        if sam_input.shape[1] == 1:
            sam_input = sam_input.repeat(1, 3, 1, 1)

        sam_input = F.interpolate(sam_input, size=(1024, 1024), mode='bilinear', align_corners=True)
        
        sam_embed = self.sam_image_encoder(sam_input)
        
        skips = self.encoder(x)

        sam_embed = F.interpolate(sam_embed, size=(skips[3].shape[2], skips[3].shape[3]), mode='bilinear', align_corners=True)
        skips[3] = torch.cat((skips[3], sam_embed), dim=1)
        

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
