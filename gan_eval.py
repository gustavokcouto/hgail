from PIL import Image
import torch as th
import numpy as np

from rl_birdview.models.ppo_policy import PpoPolicy
from rl_birdview.models.discriminator import ExpertDataset

import pathlib


if __name__ == '__main__':
    output_dir = pathlib.Path('agent_eval/gan_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = pathlib.Path('/home/casa/projects/paper_results/hgail/ckpt')
    ckpt_files = ['']
    ckpt_list = [ckpt_file for ckpt_file in ckpt_dir.iterdir() if ckpt_file.stem!='ckpt_latest']
    ckpt_list.sort(key=lambda ckpt_file: int(ckpt_file.stem.split('_')[1]))
    ckpt_idx_list = [0, 10, 20, 50, 100]
    ckpt_list = [ckpt_list[ckpt_idx] for ckpt_idx in ckpt_idx_list]

    expert_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
        ),
        batch_size=4,
        shuffle=True,
    )
    for ckpt_idx, ckpt_path in enumerate(ckpt_list):
        saved_variables = th.load(ckpt_path, map_location='cuda')
        policy_kwargs = saved_variables['policy_init_kwargs']
        policy = PpoPolicy(**policy_kwargs)
        policy.load_state_dict(saved_variables['policy_state_dict'])

        fake_birdview_tensor = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)
        birdview_list = [None for _ in range(len(expert_loader.dataset))]
        for batch in expert_loader:
            obs_dict, _ = batch
            item_idx_array = obs_dict['item_idx'].cpu().numpy()
            for item_idx in range(obs_dict['item_idx'].shape[0]):
                birdview_list[item_idx_array[item_idx]] = obs_dict['birdview'][item_idx]
        birdview_tensor = th.stack(birdview_list)
        idx_tensor = th.Tensor([89, 737, 1072, 1287, 1346]).int()
        fake_birdview_tensor = fake_birdview_tensor.index_select(dim=0, index=idx_tensor)
        birdview_tensor = birdview_tensor.index_select(dim=0, index=idx_tensor)
        img_height = 192
        for img_idx in range(len(idx_tensor)):
            fake_birdview = fake_birdview_tensor[img_idx].numpy()
            fake_birdview = np.transpose(fake_birdview, [1, 2, 0]).astype(np.uint8)
            fake_birdview = Image.fromarray(fake_birdview)

            birdview = birdview_tensor[img_idx].numpy()
            birdview = np.transpose(birdview, [1, 2, 0]).astype(np.uint8)
            birdview = Image.fromarray(birdview)

            if img_idx == 0 and ckpt_idx == 0:
                img_width = fake_birdview.size[0]
                img_height = fake_birdview.size[1]
                eval_img = Image.new("RGB", ((len(ckpt_list) + 1) * img_width, len(idx_tensor) * img_height), "white")
                birdview_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                epoch_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")

            if ckpt_idx == 0:
                birdview_img.paste(birdview, (0, img_idx * img_height))
                eval_img.paste(birdview, (0, img_idx * img_height))

            eval_img.paste(fake_birdview, ((ckpt_idx + 1) * img_width, img_idx * img_height))
            epoch_img.paste(fake_birdview, (0, img_idx * img_height))

        epoch_img.save(output_dir / '{}.png'.format(ckpt_path.stem))

    eval_img.save(output_dir / 'eval.png')
    birdview_img.save(output_dir / 'birdview.png')
