from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

import torch
import numpy as np

MAX_RESOLUTION=2048 #Who the hell feed 4k images to ControlNet?

class OpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = (["enable", "disable"], {"default": "enable"}),
            detect_body = (["enable", "disable"], {"default": "enable"}),
            detect_face = (["enable", "disable"], {"default": "enable"})
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):
        from controlnet_aux.open_pose import OpenposeDetector, PoseResult

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"


        self.openpose_json = None
        model = OpenposeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        
        def cb(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            return result[0]
        
        out = common_annotator_call(cb, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [self.openpose_json] },
            "result": (out, )
        }
    
class OpenPose_Interpolator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "from_image": ("IMAGE",),
            "batch_size" : ("INT", {"default": 2, "min": 2, "max": 128, "step": 1}),
        },
        "optional": {
            "to_image": ("IMAGE",),
            "detect_hand" : (["enable", "disable"], {"default": "enable"}),
            "detect_body" : (["enable", "disable"], {"default": "enable"}),
            "detect_face" : (["enable", "disable"], {"default": "enable"}),
            "resolution": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})
        }}
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate_poses"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def mix_keypoints(k1, k2, t):
        from controlnet_aux.open_pose import Keypoint

        # if one of the keypoints is None, return None
        if k1 is None or k2 is None:
            return None
        
        # if the ids do not match, raise an error
        if k1.id != k2.id:
            raise ValueError("The keypoints do not match!")
        
        
        # interpolate the fields x, y, score
        x = k1.x * (1 - t) + k2.x * t
        y = k1.y * (1 - t) + k2.y * t
        score = k1.score * (1 - t) + k2.score * t

        # return the interpolated keypoint
        return Keypoint(id=k1.id, x=x, y=y, score=score)

    def mix_keypoint_lists(k1, k2, t):
        by_ids = {}
        # k1 and k2 are lists where each element is either a Keypoint or None, the ids are actually always -1
        
        # enumerate the keypoints with enumerate():
        for i, k in enumerate(k1):
            by_ids[i] = [k, None]
        for i, k in enumerate(k2):
            if i not in by_ids:
                by_ids[i] = [None, k]
            else:
                [o, _] = by_ids[i]
                by_ids[i] = [o, k]

        # now we have a dict where the keys are the ids and the values are lists of length 2 with the keypoints
        res = []

        for i in range(len(by_ids)):
            [kp1, kp2] = by_ids[i]
            res.append(OpenPose_Interpolator.mix_keypoints(kp1, kp2, t))

        return res

    def mix_body_result(b1, b2, t):
        from controlnet_aux.open_pose import BodyResult
        # a body result has keypoints: List[Union[Keypoint, None]], total_score: float, total_parts: int
        # interpolate keypoints, total_score. total_parts is taken from the number of interpolated keypoints
        # if one of the bodies is None, return None
        if b1 is None or b2 is None:
            return None
        
        # interpolate the keypoints
        keypoints = OpenPose_Interpolator.mix_keypoint_lists(b1.keypoints, b2.keypoints, t)
        total_score = b1.total_score * (1 - t) + b2.total_score * t
        total_parts = len(keypoints)

        # return the interpolated body result
        return BodyResult(keypoints=keypoints, total_score=total_score, total_parts=total_parts)
    
    def mix_pose_result(p1, p2, t):
        from controlnet_aux.open_pose import PoseResult
        # a pose result has body: BodyResult, left_hand: Union[BodyResult, None], right_hand: Union[BodyResult, None], face: Union[FaceResult, None]
        # interpolate body, left_hand, right_hand, face
        # if one of the poses is None, return None
        # if one of the poses has a None hand or face, return None for that hand or face

        if p1 is None or p2 is None:
            return None
        
        body = OpenPose_Interpolator.mix_body_result(p1.body, p2.body, t)

        if p1.left_hand is None or p2.left_hand is None:
            left_hand = None
        else:
            left_hand = OpenPose_Interpolator.mix_keypoint_lists(p1.left_hand, p2.left_hand, t)

        if p1.right_hand is None or p2.right_hand is None:
            right_hand = None
        else:
            right_hand = OpenPose_Interpolator.mix_keypoint_lists(p1.right_hand, p2.right_hand, t)

        if p1.face is None or p2.face is None:
            face = None
        else:
            face = OpenPose_Interpolator.mix_keypoint_lists(p1.face, p2.face, t)

        return PoseResult(body=body, left_hand=left_hand, right_hand=right_hand, face=face)
    
    def mix_pose_result_lists(p1, p2, t):
        if len(p1) != len(p2):
            raise ValueError("The pose result lists do not match in length! Are there a different number of persons in the images?")
        
        return [
            OpenPose_Interpolator.mix_pose_result(p1[i], p2[i], t)
            for i in range(len(p1))
        ]




    def interpolate_poses(self, from_image, batch_size, detect_hand, detect_body, detect_face, to_image=None, resolution=512, **kwargs):
        from controlnet_aux.open_pose import OpenposeDetector, PoseResult, draw_poses
        from controlnet_aux.util import HWC3

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        self.openpose_json = None
        self.openpose_poses: list[PoseResult] = []
        model = OpenposeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        
        def cb(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            if len(result) > 2:
                self.openpose_poses.append(result[2])
            return result[0]
        
        # if to_image is None, we need to find an alternative
        if to_image is None:
            # if from_image is a batch, we take the last image as to_image and the first image as from_image
            if from_image.shape[0] > 1:
                # make sure to keep the rank
                to_image = from_image[-1:]
                from_image = from_image[:1]
            else:
                raise ValueError("You need to provide two images to interpolate between them!")

        
        out_from = common_annotator_call(cb, from_image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        out_to = common_annotator_call(cb, to_image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)

        # check if we have two poses
        if len(self.openpose_poses) != 2:
            raise ValueError("Could not detect two poses in the input images!")

        from_poses = self.openpose_poses[0]
        to_poses = self.openpose_poses[1]

        map_height = out_from.shape[1]
        map_width = out_from.shape[2]

        interpolated_poses = []
        pose_images = []
        for i in range(batch_size):
            t = i / (batch_size - 1)
            pose = OpenPose_Interpolator.mix_pose_result_lists(from_poses, to_poses, t)
            interpolated_poses.append(pose)
            # use OpenposeDetector to draw the interpolated poses
            canvas = draw_poses(pose, map_height, map_width, draw_body=detect_body, draw_hand=detect_hand, draw_face=detect_face)
            np_result = HWC3(canvas)
            pose_images.append(torch.from_numpy(np_result.astype(np.float32) / 255.0))

        # make a tensor that contains all the images
        out_pose_images = torch.stack(pose_images, dim=0)

        del model
        return {
            'ui': { "openpose_json": [self.openpose_json] },
            "result": (out_pose_images, )
        }

NODE_CLASS_MAPPINGS = {
    "OpenposePreprocessor": OpenPose_Preprocessor,
    "OpenposeInterpolator": OpenPose_Interpolator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposePreprocessor": "OpenPose Pose Recognition",
    "OpenposeInterpolator": "OpenPose Pose Interpolated Recognition",
}