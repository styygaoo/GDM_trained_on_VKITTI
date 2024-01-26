import os
import argparse
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from model import loader



max_depths = {
    'kitti': 80.0,
    'nyu' : 10.0,
    'nyu_reduced' : 10.0,
}
nyu_res = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}
kitti_res = {
    'full' : (384, 1280),
    'half' : (192, 640)}
resolutions = {
    'nyu' : nyu_res,
    'nyu_reduced' : nyu_res,
    'kitti' : kitti_res}
crops = {
    'kitti' : [128, 381, 45, 1196],
    'nyu' : [20, 460, 24, 616],
    'nyu_reduced' : [20, 460, 24, 616]}


test_images_path = "/HOMES/yigao/SUNrgbd/SUNRGBD-test_images"
files = [ f for f in os.listdir(test_images_path) if '.jpg' in f.lower() ]
sorted_files = sorted(files, key=lambda x: int(x.split('.')[0].split('-')[1]))




def get_args():
    parser = argparse.ArgumentParser(description='Nano Inference for Monocular Depth Estimation')

    #Mode
    parser.set_defaults(evaluate=False)

    parser.add_argument('--resolution',
                        type=str,
                        help='Resolution of the images for training',
                        choices=['full', 'half'],
                        default='full')

    #Model
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to be trained',
                        default='UpDepth')
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to model weights')
    parser.add_argument('--save_results',
                        type=str,
                        help='path to save results to',
                        default='./results')

    #System
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of dataloader workers',
                        default=2)
    return parser.parse_args()



class Inference_Engine():
    def __init__(self, args):

        self.maxDepth = 10
        self.crop = crops["nyu"]
        # self.result_dir = args.save_results
        self.result_dir = "/HOMES/yigao/SUNrgbd/predictions/"
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.device = torch.device('cuda:0')
        self.model = loader.load_model(args.model, args.weights_path)
        self.model = self.model.eval().cuda()

        self.visualize_images = [0]

        self.tensorRT_evaluate()


    def tensorRT_evaluate(self):
        torch.cuda.empty_cache()
        trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(480, 640))])
        # image = plt.imread("/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/results/image_0.png")

        for i in sorted_files:
            print(i)
            image = plt.imread("/HOMES/yigao/SUNrgbd/SUNRGBD-test_images/" + i)

            # print(image.shape)
            # print(image)
            # image = np.array(image).astype(np.float32) / 255.0
            image = trans(image)
            image = torch.clamp(image, 0.0, 1.0)
            image = image.unsqueeze(0)
            # print(image)
            # print(image.shape)
            image = image.to(self.device)
            torch.cuda.synchronize()

            inv_prediction = self.model(image)
            prediction = self.inverse_depth_norm(inv_prediction)
            # print(prediction)
            # print(prediction.shape)
            torch.cuda.synchronize()

            self.save_image_results(prediction, prediction, i)


    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth


    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth


    def save_image_results(self, image, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()      # 3,480,640 -> 480,640,3  plot image with shape (C, H, W), we need to reshape it to (H, W, C):
        # print(img)
        # print(img.shape)
        prediction = prediction[0,0].permute(0, 1).detach().cpu()
        cmap = 'viridis'

        # save_to_dir = os.path.join(self.result_dir, 'image_{}.png'.format(image_id))
        # fig = plt.figure(frameon=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(img)
        # fig.savefig(save_to_dir)
        # plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'GDM_prediction_{}'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(prediction)
        fig.savefig(save_to_dir)
        plt.clf()


if __name__ == '__main__':
    args = get_args()
    print(args)
    engine = Inference_Engine(args)
