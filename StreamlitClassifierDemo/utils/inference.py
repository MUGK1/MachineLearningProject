import torch
from .predictors import TensorFlowPredictor, PyTorchPredictor

def load_model_and_predictor(model_name, model_path):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    if model_name.startswith("efficientnet"):
        predictor = TensorFlowPredictor("efficientnetb0", model_path)
        return predictor, "tensorflow"

    elif model_name.startswith("mobilevit"):
        mv_type = "_".join(model_name.split("_")[:2])
        predictor = PyTorchPredictor(mv_type, model_path, device=device)
        return predictor, "pytorch"

    else:
        raise ValueError(f"Unsupported model name: {model_name}")