from pathlib import Path
import numpy as np
from PIL import Image


if __name__ == '__main__':
    image_path = 'gail_experts/route_00/ep_00/birdview_masks/0050_00.png'
    bev_masks_output_dir = Path('paper_plots/bev_masks')
    bev_masks_output_dir.mkdir(parents=True, exist_ok=True)
    image_array = Image.open(image_path)
    image_array = np.array(image_array, dtype=np.uint8)
    drivable_areas = image_array[:, :, 0]
    Image.fromarray(drivable_areas).save(bev_masks_output_dir / 'drivable_areas.png')
    desired_route = image_array[:, :, 1]
    Image.fromarray(desired_route).save(bev_masks_output_dir / 'desired_route.png')
    lane_boundaries = image_array[:, :, 2]
    Image.fromarray(lane_boundaries).save(bev_masks_output_dir / 'lane_boundaries.png')
