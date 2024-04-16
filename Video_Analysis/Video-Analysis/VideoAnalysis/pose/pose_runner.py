from mmpose.apis import init_model, inference_bottomup

def pose_runner(cfg, batch):
    pass

path = '/content/mmpose/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-s_8xb32-700e_crowdpose-640x640.py'

model = init_model(path, device='cpu', checkpoint = '/content/rtmo-s_8xb32-700e_crowdpose-640x640-79f81c0d_20231211.pth')
result = inference_bottomup(model, '/content/multiple-portrait-handsome-young-man-260nw-1342433660.jpg')[0]
