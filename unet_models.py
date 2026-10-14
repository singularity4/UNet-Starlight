"""unet_models.py - the U-Nets UNet-Starlight, UNet-Starlight-mini, UNet-Starlight-tiny and UNet-Radius-mini, and the
classifier and the radius regressor built from their encoder

UNet-Starlight.

Task of the U-Nets
    Segmentation of lensed arcs: for every pixel of a 101 x 101 stamp the network gives the probability that the
    pixel contains light of a lensed source. The networks are trained on the simulated images (train_simulations.py),
    where the arc mask of every lens is known.

Architecture (the same layers for all models)
    Encoder: five blocks. inc is a convolutional block; d1-d4 each consist of a 2 x 2 max pooling layer and a
    convolutional block. Max pooling rounds down, so the feature maps have 101, 50, 25, 12 and 6 pixels on a side;
    the 6 x 6 feature maps of d4 form the bottleneck.
    Decoder: four blocks u1-u4. Each enlarges its input by bilinear interpolation to the size of the encoder feature
    maps of the same level, halves the number of feature maps with a 1 x 1 convolutional layer, joins them with the
    encoder feature maps (skip connection) and applies a convolutional block.
    Convolutional block: two 3 x 3 convolutional layers, each followed by group normalisation and a ReLU activation.
    Output: a 1 x 1 convolutional layer with one value (logit) per pixel; the sigmoid of the logit is the arc
    probability.
    Spatial dropout with probability 0.1 follows the three blocks with the most parameters (d3, d4 and u1). The weights
    start from the default initialisation of PyTorch.

Reasons for these choices
    - Group normalisation is computed within each image, so it is stable for batches of 16 images and identical in
      training and in testing.
    - Interpolation to the exact size of the skip connection handles the odd image size of 101 pixels without
      cropping, and avoids the checkerboard patterns of transposed convolutions on the scale of arcs.
    - The 6 x 6 bottleneck sees the whole stamp, and therefore every plausible Einstein radius, while the skip
      connections keep the width of arcs of one to three pixels.
    - The 3 x 3 convolutional layers have no bias, because the group normalisation that follows has its own shift.

Models
    The models differ only in the number of filters b of the first convolutional block, which doubles after every
    pooling layer up to 16 b in the bottleneck. The number of weights of a convolutional layer is proportional to the
    product of its numbers of input and output feature maps, so halving b divides the number of parameters by about
    four.
        starlight        UNet-Starlight        b = 16   1,811,729 parameters
        starlight-mini   UNet-Starlight-mini   b =  8     453,769 parameters
        starlight-tiny   UNet-Starlight-tiny   b =  4     113,861 parameters
        radius-mini      UNet-Radius-mini      b =  8     453,898 parameters
    The first column is the key of a model (option --model of the scripts, and the name of its results folders). The
    earlier keys sparkle, twinkle and sparkle-er are still accepted for starlight-mini, starlight-tiny and
    radius-mini.

UNet-Radius-mini
    UNet-Starlight-mini with a second output, the Einstein radius in arcseconds: global average pooling of the 128
    bottleneck feature maps, followed by a linear layer with one output (129 parameters). The segmentation layers are
    unchanged.
    The linear layer is created after all other layers, so that for a given seed the other layers start from the same
    random weights as in UNet-Starlight-mini.

Classifier for the real benchmarks
    The five encoder blocks of a U-Net, global average pooling of the bottleneck feature maps and a new linear layer
    with two outputs (non-lens, lens); the lens probability is the softmax of the two outputs. The decoder is not used.
    The encoder starts either from the weights of the U-Net trained on simulations (pretrained) or from random weights;
    a third option, a fixed pretrained encoder (frozen), was used in the initial experiments.

Radius regressor for the real lenses
    The five encoder blocks of UNet-Radius-mini, global average pooling and its Einstein radius layer. It is
    fine-tuned on the measured Einstein radii of the real lenses (einstein_radius.py), starting from the weights of
    UNet-Radius-mini trained on simulations or from random weights.

Saved networks
    A saved network stores the constructor arguments of UNet ("architecture": in_channels, base, dropout and, for
    UNet-Radius-mini, einstein_radius) and the weights. The weights are stored under the names of the network parts
    (inc, d1-d4, u1-u4, out, radius; within the blocks net, reduce and conv); these names must not change, or saved
    networks can no longer be loaded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FILTERS = {"starlight": 16, "starlight-mini": 8, "starlight-tiny": 4, "radius-mini": 8}   # filters of the first block
NAMES = {"starlight": "UNet-Starlight", "starlight-mini": "UNet-Starlight-mini",
         "starlight-tiny": "UNet-Starlight-tiny", "radius-mini": "UNet-Radius-mini"}
EINSTEIN_RADIUS_MODELS = ("radius-mini",)         # models with the Einstein radius output
EARLIER_KEYS = {"sparkle": "starlight-mini", "twinkle": "starlight-tiny",
                "sparkle-er": "radius-mini"}                 # earlier keys, still accepted
ENCODER_BLOCKS = ("inc", "d1", "d2", "d3", "d4")


# ================================================================================================ 1. network parts

def number_of_groups(feature_maps):
    """Number of groups of the group normalisation: the largest of 8, 4, 2 and 1 that divides the number of feature
    maps."""
    for groups in (8, 4, 2, 1):
        if feature_maps % groups == 0:
            return groups
    return 1


class ConvolutionalBlock(nn.Module):
    """Two 3 x 3 convolutional layers, each followed by group normalisation and a ReLU activation; optionally spatial
    dropout, which sets whole feature maps to zero."""

    def __init__(self, input_maps, output_maps, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(input_maps, output_maps, 3, padding=1, bias=False),
                  nn.GroupNorm(number_of_groups(output_maps), output_maps),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(output_maps, output_maps, 3, padding=1, bias=False),
                  nn.GroupNorm(number_of_groups(output_maps), output_maps),
                  nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


class EncoderBlock(nn.Module):
    """Encoder block d1-d4: a 2 x 2 max pooling layer, which halves the size of the feature maps (rounding down),
    followed by a convolutional block."""

    def __init__(self, input_maps, output_maps, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvolutionalBlock(input_maps, output_maps, dropout))

    def forward(self, features):
        return self.net(features)


class DecoderBlock(nn.Module):
    """Decoder block u1-u4: bilinear interpolation to the size of the skip connection, a 1 x 1 convolutional layer
    that halves the number of feature maps, joining with the feature maps of the skip connection, and a convolutional
    block."""

    def __init__(self, input_maps, skip_maps, output_maps, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv2d(input_maps, output_maps, 1)
        self.conv = ConvolutionalBlock(output_maps + skip_maps, output_maps, dropout)

    def forward(self, features, skip_features):
        features = F.interpolate(features, size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.reduce(features)
        features = torch.cat([skip_features, features], dim=1)
        return self.conv(features)


# ===================================================================================================== 2. U-Net

class UNet(nn.Module):
    """U-Net with `base` filters in the first convolutional block. Output: one logit per pixel; with
    einstein_radius=True, also the estimated Einstein radius from the bottleneck feature maps."""

    def __init__(self, in_channels=1, base=16, dropout=0.10, einstein_radius=False):
        super().__init__()
        self.bottleneck_maps = 16 * base
        self.inc = ConvolutionalBlock(in_channels, base)
        self.d1 = EncoderBlock(base, 2 * base)
        self.d2 = EncoderBlock(2 * base, 4 * base)
        self.d3 = EncoderBlock(4 * base, 8 * base, dropout=dropout)
        self.d4 = EncoderBlock(8 * base, 16 * base, dropout=dropout)
        self.u1 = DecoderBlock(16 * base, 8 * base, 8 * base, dropout=dropout)
        self.u2 = DecoderBlock(8 * base, 4 * base, 4 * base)
        self.u3 = DecoderBlock(4 * base, 2 * base, 2 * base)
        self.u4 = DecoderBlock(2 * base, base, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.einstein_radius = bool(einstein_radius)
        if self.einstein_radius:        # created last: the other layers keep the random weights of UNet-Starlight-mini
            self.radius = nn.Linear(16 * base, 1)

    def forward(self, image, return_einstein_radius=False):
        """Logits of the arc probability of every pixel; with return_einstein_radius=True, also the estimated
        Einstein radius (arcseconds) of every image."""
        level_0 = self.inc(image)              # 101 x 101
        level_1 = self.d1(level_0)             # 50 x 50
        level_2 = self.d2(level_1)             # 25 x 25
        level_3 = self.d3(level_2)             # 12 x 12
        bottleneck = self.d4(level_3)          # 6 x 6
        features = self.u1(bottleneck, level_3)
        features = self.u2(features, level_2)
        features = self.u3(features, level_1)
        features = self.u4(features, level_0)
        logits = self.out(features)
        if return_einstein_radius:
            if not self.einstein_radius:
                raise ValueError("This U-Net has no Einstein radius output")
            return logits, self.radius(bottleneck.mean(dim=(2, 3))).squeeze(1)
        return logits


def model_key(name):
    """The key of a model; an earlier key (sparkle, twinkle, sparkle-er) is replaced by the current one."""
    return EARLIER_KEYS.get(name, name)


def build_unet(model, dropout=0.10):
    """A new U-Net with random weights; model is starlight, starlight-mini, starlight-tiny or radius-mini."""
    return UNet(base=FILTERS[model], dropout=dropout, einstein_radius=model in EINSTEIN_RADIUS_MODELS)


def architecture_of(model, dropout=0.10):
    """The constructor arguments of UNet for a model, as stored with a saved network."""
    architecture = {"in_channels": 1, "base": FILTERS[model], "dropout": dropout}
    if model in EINSTEIN_RADIUS_MODELS:
        architecture["einstein_radius"] = True
    return architecture


# ====================================================================== 3. classifier and radius regressor

class EncoderClassifier(nn.Module):
    """The encoder of a U-Net, global average pooling of the bottleneck feature maps and a linear layer with two
    outputs (non-lens, lens)."""

    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.ModuleDict({block: getattr(unet, block) for block in ENCODER_BLOCKS})
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(unet.bottleneck_maps, 2)

    def features(self, images):
        """The bottleneck feature maps, averaged over the 6 x 6 pixels."""
        feature_maps = images
        for block in ENCODER_BLOCKS:
            feature_maps = self.encoder[block](feature_maps)
        return self.pooling(feature_maps).flatten(1)

    def forward(self, images):
        return self.classifier(self.features(images))


def build_classifier(saved_network, initialisation):
    """The classifier with the architecture of a saved U-Net. initialisation:
        pretrained   the encoder starts from the weights of the saved network, and all weights are trained;
        random       the encoder starts from random weights;
        frozen       the encoder has the weights of the saved network and is kept fixed during training, so that
                     only the linear layer is trained (the training script keeps it fixed). Used in the initial
                     experiments, not in the results of the paper.
    The linear layer starts from random weights in all three cases. The complete U-Net, decoder included, is built
    before the linear layer, so that for a given seed the random weights are the same in all three cases."""
    if initialisation not in ("pretrained", "random", "frozen"):
        raise ValueError(f"Unknown initialisation: {initialisation}")
    unet = UNet(**saved_network["architecture"])
    if initialisation in ("pretrained", "frozen"):
        unet.load_state_dict(saved_network["model_state_dict"])
    return EncoderClassifier(unet)


class RadiusRegressor(nn.Module):
    """The encoder of UNet-Radius-mini, global average pooling of the bottleneck feature maps and its Einstein radius
    layer: the network that is fine-tuned on the measured radii of the real lenses. The decoder is not used, because
    the real stamps have no arc masks."""

    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.ModuleDict({block: getattr(unet, block) for block in ENCODER_BLOCKS})
        self.radius = unet.radius

    def forward(self, images):
        feature_maps = images
        for block in ENCODER_BLOCKS:
            feature_maps = self.encoder[block](feature_maps)
        return self.radius(feature_maps.mean(dim=(2, 3))).squeeze(1)


def build_regressor(saved_network, pretrained):
    """The radius regressor with the architecture of a saved UNet-Radius-mini. pretrained=True: encoder and radius layer
    start from the weights of the saved network; pretrained=False: from random weights."""
    unet = UNet(**saved_network["architecture"])
    if pretrained:
        unet.load_state_dict(saved_network["model_state_dict"])
    return RadiusRegressor(unet)


# ================================================================================================ 4. saved networks

def load_saved_network(path):
    """A network saved by train_simulations.py: weights, architecture, input transform and decision rule."""
    return torch.load(path, map_location="cpu")


def model_of_saved_network(saved_network):
    """starlight, starlight-mini, starlight-tiny or radius-mini, from the number of filters of the first convolutional
    block and the presence of the Einstein radius output."""
    base = int(saved_network["architecture"].get("base", 16))
    if saved_network["architecture"].get("einstein_radius", False):
        return "radius-mini"
    for model, filters in FILTERS.items():
        if filters == base and model not in EINSTEIN_RADIUS_MODELS:
            return model
    raise ValueError(f"No model has {base} filters in the first convolutional block")


def choose_device(name="auto"):
    """The processor on which the networks run: cpu, cuda (a GPU), or auto (a GPU if one is available)."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was requested, but none is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
