import os
import yaml

from cam_to_target_trans import cam_to_target_trans, demo_test_detection
from func_point_transfer import func_point_transfer

from local_services.owlv2_local import OWLViT
from local_services.sam_local import SAM


def run_functo_pipeline():
    # Hardcoded config
    config = {
        "task_label": "pour",
        "test_tool_label": "teapot",
        "test_target_label": "bowl",
        "params": {
            "num_candidate_keypoints": 12,
            "vp_flag": False,
            "sd_dino_flag": False
        },
        "owlv2": {
            "box_threshold": 0.10,
            "text_threshold": 0.10,
            "conf_threshold": 0.05
        },
        "paths": {
            "base_data_path": "/home/oguz/Desktop/attempt_three"
        }
    }

    # Initialize detectors
    owl = OWLViT()
    sam_predictor = SAM(checkpoint="/home/oguz/Desktop/attempt_three/segment-anything/weights/sam_vit_h_4b8939.pth")

    # Load config fields
    task_label = config['task_label']
    test_tool_label = config['test_tool_label']
    test_target_label = config['test_target_label']
    num_candidate_keypoints = config['params']['num_candidate_keypoints']
    vp_flag = config['params']['vp_flag']
    sd_dino_flag = config['params']['sd_dino_flag']
    base_data_path = config['paths']['base_data_path']
    data_path = os.path.join(base_data_path, 'demo_data', f'{task_label}_demo')
    test_data_path = os.path.join(base_data_path, 'test_data', f'{task_label}_test')
    owl_params = config.get("owlv2", {})

    # Step 1: Detect and segment tool/target
    print("############ Test target object frame transformation ############")
    print("Using text prompt:", test_tool_label)
    demo_test_detection(
        test_data_path,
        test_tool_label,
        target_label=test_target_label,
        owl_params=owl_params,
        owl=owl,
        sam_predictor=sam_predictor
    )

    cam_to_target_trans(test_data_path)

    # Step 2: Transfer functional point
    print("############ Function point transfer ############")
    keypoints_tensor = func_point_transfer(
        data_path,
        test_data_path,
        test_tool_label,
        test_target_label,
        task_label,
        num_candidate_keypoints,
        sd_dino_flag
    )

    return keypoints_tensor

if __name__ == '__main__':
    keypoints = run_functo_pipeline()
    print("Extracted keypoints shape:", keypoints.shape)