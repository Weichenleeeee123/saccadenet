"""Candidate views reconstructed solely from a completed retina sample."""

import math

import cv2
import numpy as np

from saccadenet.contracts import RetinaOut


def candidate_view(retina: RetinaOut, xy: tuple[float, float], size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(size, dtype=np.float32) - size / 2
    gx, gy = np.meshgrid(row + xy[0], row + xy[1])
    dx = gx - retina.fixation_xy[0]
    dy = gy - retina.fixation_xy[1]
    radius = np.hypot(dx, dy)
    sectors = retina.logpolar.shape[1]
    delta = 2 * math.pi / sectors
    radial_index = np.log(np.maximum(radius, retina.fovea.shape[0] / 2) / (retina.fovea.shape[0] / 2)) / delta
    angular_index = np.mod(np.arctan2(dy, dx), 2 * math.pi) / delta
    angular_index = np.where(angular_index >= sectors, 0, angular_index)
    # Extend the first sector so bilinear interpolation wraps at 2π.
    strip = np.concatenate((retina.logpolar, retina.logpolar[:, :1]), axis=1)
    strip_valid = np.concatenate((retina.logpolar_valid, retina.logpolar_valid[:, :1]), axis=1).astype(np.float32)
    outer = cv2.remap(strip, angular_index.astype(np.float32), radial_index.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    outer_valid = cv2.remap(strip_valid, angular_index.astype(np.float32), radial_index.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0.999

    fovea_x = (dx + retina.fovea.shape[1] / 2).astype(np.float32)
    fovea_y = (dy + retina.fovea.shape[0] / 2).astype(np.float32)
    inner = cv2.remap(retina.fovea, fovea_x, fovea_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inner_valid = cv2.remap(retina.fovea_valid.astype(np.uint8), fovea_x, fovea_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    in_fovea = (fovea_x >= 0) & (fovea_x < retina.fovea.shape[1]) & (fovea_y >= 0) & (fovea_y < retina.fovea.shape[0])
    view = np.where(in_fovea[..., None], inner, outer)
    valid = np.where(in_fovea, inner_valid, outer_valid)
    height, width = retina.canvas_shape
    valid &= (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    return view.astype(np.uint8), valid
