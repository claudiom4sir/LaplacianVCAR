import os


def get_video_info(vname):
    # 3fjords-1_1920x1080_880.yuv
    name = vname[:vname.find('_')]
    vname = vname[len(name)+1:]
    w = vname[:vname.find('x')]
    vname = vname[len(w)+1:]
    h = vname[:vname.find('_')]
    return name, w, h

roots = ['train_108/HM16.5_LDP/QP42/', 'train_108/HM16.5_LDP/QP32/', 'train_108/HM16.5_LDP/QP27/', 'train_108/HM16.5_LDP/QP22/']
targets = ['MFQEv2_RGB/train_108/HM16.5_LDP/QP42/', 'MFQEv2_RGB/train_108/HM16.5_LDP/QP32/', 'MFQEv2_RGB/train_108/HM16.5_LDP/QP27/', 'MFQEv2_RGB/train_108/HM16.5_LDP/QP22/']
for i in range(len(roots)):
    root = roots[i]
    target = targets[i]
    os.makedirs(target, exist_ok=True)
    videos = os.listdir(root)
    for video in videos:
        if '.yuv' not in video:
            continue
        name, w, h = get_video_info(video)
        video_name = video.replace('.yuv', '')
        video_path_old = os.path.join(root, video)
        video_path_new = os.path.join(target, video_name)
        os.makedirs(video_path_new, exist_ok=True)
        cmd = f'ffmpeg -pix_fmt yuv420p -s {w}x{h} -i {video_path_old} {video_path_new}'
        os.system(cmd)
