import os
import torch

__all__ = ['restore']

def restore(args, model, optimizer, istrain=True, including_opt=False):
    if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
        snapshot = args.restore_from
    else:
        restore_dir = args.snapshot_dir
        filelist = os.listdir(restore_dir)
        filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
        if len(filelist) > 0:
            filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
            snapshot = os.path.join(restore_dir, filelist[0])
        else:
            snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            if istrain:
                args.current_epoch = checkpoint['epoch'] + 1
                args.global_counter = checkpoint['global_counter'] + 1
                if including_opt:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(snapshot, checkpoint['epoch']))
        except KeyError:
            print "KeyError"
            if args.arch=='vgg_v5_7' or args.arch=='vgg_v7' or args.arch=='vgg_v10':
                _model_load_v6(model, checkpoint)
            # elif args.arch=='vgg_v2':
            #     _model_load_v2(model, checkpoint)
            else:
                _model_load(model, checkpoint)
        except KeyError:
            print "Loading pre-trained values failed."
            raise
        print("=> loaded checkpoint '{}'".format(snapshot))
    else:
        print("=> no checkpoint found at '{}'".format(snapshot))


def _model_load(model, pretrained_dict):
    model_dict = model.state_dict()

    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if model_dict.keys()[0].startswith('module.'):
        pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}

    # print pretrained_dict.keys()
    # print model.state_dict().keys()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print "Weights cannot be loaded:"
    print [k for k in model_dict.keys() if k not in pretrained_dict.keys()]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def _model_load_v6(model, pretrained_dict):
    model_dict = model.state_dict()

    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if model_dict.keys()[0].startswith('module.'):
        pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}


    feature2_pred_w = {'module.fc5_seg.%d.weight'%(i):'module.features.%d.weight'%(i+24) for i in range(0,5,2)}
    feature2_pred_b = {'module.fc5_seg.%d.bias'%(i):'module.features.%d.bias'%(i+24) for i in range(0,5,2)}
    # feature_erase_pred_w = {'module.fc5_seg.%d.weight'%(i):'module.features.%d.weight'%(i+24) for i in range(0,5,2)}
    # feature_erase_pred_b = {'module.fc5_seg.%d.bias'%(i):'module.features.%d.bias'%(i+24) for i in range(0,5,2)}

    common_pred = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print "Weights cannot be loaded:"
    print [k for k in model_dict.keys() if k not in common_pred.keys()+ feature2_pred_w.keys() + feature2_pred_b.keys()]

    def update_coord_dict(d):
        for k in d.keys():
            model_dict[k] = pretrained_dict[d[k]]

    update_coord_dict(feature2_pred_w)
    update_coord_dict(feature2_pred_b)
    # update_coord_dict(feature_erase_pred_w)
    # update_coord_dict(feature_erase_pred_b)


    model_dict.update(common_pred)
    model.load_state_dict(model_dict)

def _model_load_v2(model, pretrained_dict):
    model_dict = model.state_dict()

    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if model_dict.keys()[0].startswith('module.'):
        pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}


    fc5_cls_w = {'module.fc5_cls.%d.weight'%(i):'module.features.%d.weight'%(i+24) for i in range(0,5,2)}
    fc5_cls_b = {'module.fc5_cls.%d.bias'%(i):'module.features.%d.bias'%(i+24) for i in range(0,5,2)}
    fc5_seg_w = {'module.fc5_seg.%d.weight'%(i):'module.features.%d.weight'%(i+24) for i in range(0,5,2)}
    fc5_seg_b = {'module.fc5_seg.%d.bias'%(i):'module.features.%d.bias'%(i+24) for i in range(0,5,2)}

    common_pred = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print "Weights cannot be loaded:"
    print [k for k in model_dict.keys() if k not in common_pred.keys()+fc5_cls_w.keys()+
           fc5_cls_b.keys() + fc5_seg_w.keys() + fc5_seg_b.keys()]

    def update_coord_dict(d):
        for k in d.keys():
            model_dict[k] = pretrained_dict[d[k]]

    update_coord_dict(fc5_cls_w)
    update_coord_dict(fc5_cls_b)
    update_coord_dict(fc5_seg_w)
    update_coord_dict(fc5_seg_b)


    model_dict.update(common_pred)
    model.load_state_dict(model_dict)
