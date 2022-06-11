import architectures.resnet50 as resnet50
import architectures.googlenet as googlenet
import architectures.bninception as bninception
import architectures.distilbert as distilbert

def select(arch, opt):
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
    if 'distilbert_all' in arch:
        return distilbert.Network(opt, True)
    if 'distilbert_first' in arch:
        return distilbert.Network(opt, False)
