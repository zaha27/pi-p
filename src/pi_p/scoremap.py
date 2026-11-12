import cv2
import numpy as np
import torch


class ScoreMap:
    def __init__(self, net: torch.nn.Module):
        layers = net.model
        head = layers[-1]
        self.logits = {}
        self.input_hw = None
        layers[0].register_forward_pre_hook(self._capture_input)
        for branch in ("cv3", "one2one_cv3"):
            for i, conv in enumerate(getattr(head, branch, None) or []):
                conv.register_forward_hook(self._capture(branch, i))

    def _capture_input(self, _module, args):
        self.input_hw = tuple(args[0].shape[-2:])
        self.logits.clear()

    def _capture(self, branch: str, i: int):
        def hook(_module, _args, out):
            self.logits[(branch, i)] = out[0].detach()
        return hook

    def heatmap(self, h: int, w: int) -> np.ndarray:
        branch = "one2one_cv3" if any(b == "one2one_cv3" for b, _ in self.logits) else "cv3"
        ih, iw = self.input_hw
        scales = [v.float().sigmoid().amax(0, keepdim=True)[None] for (b, _), v in sorted(self.logits.items()) if b == branch]
        prob = torch.stack([torch.nn.functional.interpolate(s, (ih, iw), mode="bilinear") for s in scales]).amax(0)[0, 0]
        prob = prob.cpu().numpy()
        r = min(ih / h, iw / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (ih - nh) // 2, (iw - nw) // 2
        return cv2.resize(prob[top : top + nh, left : left + nw], (w, h))
