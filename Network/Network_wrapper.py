from Network.UDBRNet import UDBRNet

def model_wrapper(conf):
    in_channel = 1
    if (conf.dataset == "LCTSC"):
        out_channel = 6
    elif (conf.dataset == "SegThor"):
        out_channel = 5

    architecture = {
        "UDBRNet": UDBRNet(in_channels=in_channel, out_channels=out_channel),
    }

    model = architecture[conf.model_name]

    return model