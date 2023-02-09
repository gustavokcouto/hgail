import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import torch as th

from gym.wrappers.monitoring.video_recorder import ImageEncoder
import torchvision.transforms as transforms
from stable_baselines3.common.vec_env.base_vec_env import tile_images


def process_image(image_path):
    image_array = Image.open(image_path)
    image_array = np.array(image_array)
    image_array = np.transpose(image_array, [2, 0, 1])
    image_tensor = th.as_tensor(image_array)
    image_transforms = transforms.Compose([
        transforms.Resize((192, 192))
    ])
    image_tensor = image_transforms(image_tensor)
    image_array = image_tensor.numpy()
    image_array = np.transpose(image_array, [1, 2, 0])
    return image_array


def get_eval_data(eval_set):
    dataset_path = Path('agent_eval')
    eval_images = []
    for route_idx in [eval_set * 2, eval_set * 2 + 1]:
        route_path = dataset_path / ('route_%02d' % route_idx)
        route_df = pd.read_json(route_path / 'episode.json')
        traj_length = route_df.shape[0]
        traj_images = []
        for step_idx in range(traj_length):
            step_birdview = process_image(route_path / 'birdview_masks/{:0>4d}_00.png'.format(step_idx))
            image_height, image_width, _ = step_birdview.shape
            central_rgb = process_image(route_path / 'central_rgb/{:0>4d}.png'.format(step_idx))
            left_rgb = process_image(route_path / 'left_rgb/{:0>4d}.png'.format(step_idx))
            right_rgb = process_image(route_path / 'right_rgb/{:0>4d}.png'.format(step_idx))

            eval_image = np.zeros([image_height, 4 * image_width, 3], dtype=np.uint8)
            eval_image[:image_height, :image_width] = step_birdview
            eval_image[:image_height, image_width:2*image_width] = left_rgb
            eval_image[:image_height, 2*image_width:3*image_width] = central_rgb
            eval_image[:image_height, 3*image_width:4*image_width] = right_rgb
            traj_images.append(eval_image)

        eval_images.append(traj_images)

    max_len = 0
    for traj_images in eval_images:
        if max_len < len(traj_images):
            max_len = len(traj_images)
    
    eval_images_dataset = []
    for _ in range(max_len):
        eval_images_dataset.append([])

    for traj_images in eval_images:
        for image_idx in range(max_len):
            traj_image_idx = min(image_idx, len(traj_images)-1)
            eval_images_dataset[image_idx].append(traj_images[traj_image_idx])

    eval_dataset = []
    for im_trajs in eval_images_dataset:
        step_image = tile_images(im_trajs)
        eval_dataset.append(step_image)

    return eval_dataset


if __name__ == '__main__':
    _video_path = Path('video')
    _video_path.mkdir(parents=True, exist_ok=True)

    n_eval_sets = 15
    for i_eval_set in range(n_eval_sets):
        eval_dataset = get_eval_data(i_eval_set)
        buffer_video_path = (_video_path / 'eval_buffer_{}.mp4'.format(i_eval_set)).as_posix()
        encoder = ImageEncoder(buffer_video_path, eval_dataset[0].shape, 30, 30)
        for im in eval_dataset:
            encoder.capture_frame(im)
        encoder.close()
        encoder = None
