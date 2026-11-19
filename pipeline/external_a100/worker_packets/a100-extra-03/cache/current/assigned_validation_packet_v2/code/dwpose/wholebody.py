# https://github.com/IDEA-Research/DWPose
from pathlib import Path
import os

import numpy as np
import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose, inference_pose_batch

ModelDataPathPrefix = Path("./pretrained_weights")


class Wholebody:
    def __init__(self, device, use_batch=False):
        allow_cpu = os.environ.get("DWPOSE_ALLOW_CPU", "0") == "1"
        if hasattr(ort, "preload_dlls"):
            # Resolve CUDA/cuDNN from the active Python environment instead of
            # a hard-coded python3.x/site-packages path.
            ort.preload_dlls(directory="")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available and not allow_cpu:
            raise RuntimeError(
                "onnxruntime-gpu does not expose CUDAExecutionProvider; "
                f"available={available}. Set DWPOSE_ALLOW_CPU=1 only for diagnostics."
            )
        cuda_options = {
            "device_id": int(device),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }
        providers = [("CUDAExecutionProvider", cuda_options)]
        if allow_cpu:
            providers.append("CPUExecutionProvider")
        onnx_det = ModelDataPathPrefix.joinpath("DWPose/yolox_l.onnx")
        onnx_pose = ModelDataPathPrefix.joinpath("DWPose/dw-ll_ucoco_384.onnx")

        missing = [str(path) for path in (onnx_det, onnx_pose) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing DWPose ONNX weights: {missing}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session_det = ort.InferenceSession(
            path_or_bytes=onnx_det, sess_options=session_options, providers=providers
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=onnx_pose, sess_options=session_options, providers=providers
        )
        active = {
            "detector": self.session_det.get_providers(),
            "pose": self.session_pose.get_providers(),
        }
        if not allow_cpu and any("CUDAExecutionProvider" not in value for value in active.values()):
            raise RuntimeError(
                "DWPose CUDA initialization failed and ONNX Runtime fell back to CPU: "
                f"active={active}. Check the NVIDIA driver and CUDA/cuDNN packages "
                "installed in the active Python environment."
            )
        print(
            "DWPose ONNX providers: "
            f"detector={active['detector']} pose={active['pose']} device={device}",
            flush=True,
        )
        self.use_batch = use_batch

    def _get_result_from_det_pose(self, det_result, keypoints, scores):
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)    # (1, 133, 3)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)   # (1, 3)，对第五第六个点做平均
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)   # 从第二个开始切片，这里维度为3，只切一片
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)    # 在17索引处插入neck
        # 调换骨骼索引
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]     # openpose需要检测17点+1脖子关键点
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores, det_result


    def __call__(self, oriImg):
        if not self.use_batch:
            det_result = inference_detector(self.session_det, oriImg)
            keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)   # keypoints: (n_bbox, 133, 2) scores: (n_bbox, 133)  不管输入是什么初步提取的关键点数量都是一致的
            return self._get_result_from_det_pose(det_result=det_result, keypoints=keypoints, scores=scores)
            
        else:
            images = list(oriImg) if isinstance(oriImg, (list, tuple)) else [oriImg]
            det_results = []
            for image in images:
                boxes = inference_detector(self.session_det, image)
                boxes_array = np.asarray(boxes, dtype=np.float32)
                if boxes_array.size == 0:
                    boxes_array = np.empty((0, 4), dtype=np.float32)
                det_results.append(boxes_array.reshape(-1, 4))
            keypoints_batch, scores_batch = inference_pose_batch(
                self.session_pose, det_results, images
            )
            return [
                self._get_result_from_det_pose(det_result, keypoints, scores)
                for det_result, keypoints, scores in zip(
                    det_results, keypoints_batch, scores_batch
                )
            ]
