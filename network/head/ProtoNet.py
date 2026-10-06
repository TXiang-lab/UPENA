import torch.nn as nn


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a Config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_cfg):
        nonlocal in_channels

        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.

        num_channels = layer_cfg[0]
        kernel_size = layer_cfg[1]


        layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]
    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels


def proto_net(channels):
    return make_net(channels, [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1,'stride':2}),
                                      (256, 3, {'padding': 1}), (32, 1, {})],True)
