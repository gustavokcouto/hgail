
from rl_birdview.models.gan_layers import GanFakeBirdview
from rl_birdview.models.discriminator import ExpertDataset
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    expert_loader = torch.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=10,
            n_eps=1,
        ),
        batch_size=256,
        shuffle=True,
    )
    gan_fake_birdview = GanFakeBirdview()
    generator_variables = torch.load('saved_models/fake_birdview/generator_8.pth', map_location='cuda')
    gan_fake_birdview.generator.load_state_dict(generator_variables)
    fake_birdview_tensor = gan_fake_birdview.fill_expert_dataset(expert_loader)
    route_id = 0
    ep_id = 0
    i_step = 0
    expert_file_dir = Path('gail_experts')
    traj_length = 0
    for img_idx in range(fake_birdview_tensor.shape[0]):
        if i_step >= traj_length:
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'fake_birdview').mkdir(parents=True)
            route_df = pd.read_json(episode_dir / 'episode.json')
            traj_length = route_df.shape[0]
            route_id += 1
            i_step = 0

        fake_birdview = fake_birdview_tensor[img_idx].numpy()
        fake_birdview = np.transpose(fake_birdview, [1, 2, 0]).astype(np.uint8)
        Image.fromarray(fake_birdview).save(episode_dir / 'fake_birdview' / '{:0>4d}.png'.format(i_step))
        i_step += 1
