# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under Apache-2.0; see vendor/samurai/LICENSE.
"""SAMURAI mask-head implementation with batched host-device transfers.

Based on ``SAM2Base._forward_sam_heads`` in
``vendor/samurai/sam2/modeling/sam2_base.py``; source and license attribution
are recorded in THIRD_PARTY_NOTICES.md and the vendored closure.
Kalman boxes are reduced on the GPU and read back in one transfer per decision,
preserving SAMURAI's integer and comparison conventions. The streaming
predictor selects these heads through the upstream model factory.
"""
from __future__ import annotations


def boxes_xyxy(masks):
    """``[M, 4]`` int64 ``[x_min, y_min, x_max, y_max]`` over ``masks > 0`` for ``[M, H, W]``
    logits; an empty mask reads ``[0, 0, 0, 0]``, SAMURAI's convention."""
    import torch

    positive = masks > 0.0
    rows, cols = positive.any(dim=2).int(), positive.any(dim=1).int()
    h, w = positive.shape[1:]
    boxes = torch.stack([cols.argmax(dim=1), rows.argmax(dim=1),
                         (w - 1) - cols.flip(1).argmax(dim=1),
                         (h - 1) - rows.flip(1).argmax(dim=1)], dim=1)
    return torch.where(positive.flatten(1).any(dim=1, keepdim=True), boxes,
                       torch.zeros_like(boxes))


def _prompt_embeddings(self, batch, device, coords, labels, mask_prompt, *, memo):
    """The prompt encoder's (sparse, dense) pair. With MEMO, the empty prompt every tracked
    frame sends is cached per batch size, device and autocast dtype."""
    import torch

    if not memo:
        return self.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=mask_prompt)
    autocast = torch.is_autocast_enabled()
    key = (batch, str(device), torch.get_autocast_dtype("cuda") if autocast else None)
    cache = self._empty_prompt_cache
    if key not in cache:
        cache[key] = self.sam_prompt_encoder(points=(coords, labels), boxes=None,
                                             masks=mask_prompt)
    return cache[key]


def forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None,
                      high_res_features=None, multimask_output=False):
    """SAMURAI's ``SAM2Base._forward_sam_heads`` (vendor/samurai/sam2/modeling/sam2_base.py)
    with the same returns and side effects."""
    import torch
    import torch.nn.functional as F
    from sam2.modeling.sam2_base import NO_OBJ_SCORE

    B = backbone_features.size(0)
    device = backbone_features.device
    assert backbone_features.size(1) == self.sam_prompt_embed_dim
    assert backbone_features.size(2) == self.sam_image_embedding_size
    assert backbone_features.size(3) == self.sam_image_embedding_size

    if point_inputs is not None:
        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
        assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
    else:
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

    if mask_inputs is not None:
        assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
        if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
            sam_mask_prompt = F.interpolate(
                mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size,
                align_corners=False, mode="bilinear", antialias=True)
        else:
            sam_mask_prompt = mask_inputs
    else:
        sam_mask_prompt = None

    sparse_embeddings, dense_embeddings = _prompt_embeddings(
        self, B, device, sam_point_coords, sam_point_labels, sam_mask_prompt,
        memo=point_inputs is None and mask_inputs is None)
    low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
        image_embeddings=backbone_features,
        image_pe=self.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    if self.pred_obj_scores:
        is_obj_appearing = object_score_logits > self.min_obj_score_logits
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks,
                                         NO_OBJ_SCORE)
    low_res_multimasks = low_res_multimasks.float()
    high_res_multimasks = F.interpolate(low_res_multimasks,
                                        size=(self.image_size, self.image_size),
                                        mode="bilinear", align_corners=False)

    sam_output_token = sam_output_tokens[:, 0]
    kf_ious = None
    if multimask_output and self.samurai_mode:
        batch_inds = torch.arange(B, device=device)
        threshold = self.stable_ious_threshold
        if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
            best_iou_inds = torch.argmax(ious, dim=-1)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_bbox = boxes_xyxy(high_res_masks[0])[0].tolist()
            self.kf_mean, self.kf_covariance = self.kf.initiate(
                self.kf.xyxy_to_xyah(high_res_bbox))
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            self.frame_cnt += 1
            self.stable_frames += 1
        elif self.stable_frames < self.stable_frames_threshold:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            best_iou_inds = torch.argmax(ious, dim=-1)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            stable = (ious[0][best_iou_inds] > threshold).reshape(-1).long()
            *high_res_bbox, is_stable = torch.cat(
                [boxes_xyxy(high_res_masks[0])[0], stable]).tolist()
            if is_stable:
                self.kf_mean, self.kf_covariance = self.kf.update(
                    self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                self.stable_frames += 1
            else:
                self.stable_frames = 0
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            self.frame_cnt += 1
        else:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            high_res_multibboxes = boxes_xyxy(high_res_multimasks[0]).tolist()
            kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes))
            if device.type == "cuda":
                kf_ious = kf_ious.pin_memory()
            kf_ious = kf_ious.to(device, non_blocking=True)
            weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
            best_iou_inds = torch.argmax(weighted_ious, dim=-1)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            self.frame_cnt += 1
            unstable = (ious[0][best_iou_inds] < threshold).reshape(-1).long()
            is_unstable, best = torch.cat([unstable, best_iou_inds.reshape(-1)]).tolist()
            if is_unstable:
                self.stable_frames = 0
            else:
                self.kf_mean, self.kf_covariance = self.kf.update(
                    self.kf_mean, self.kf_covariance,
                    self.kf.xyxy_to_xyah(high_res_multibboxes[best]))
    elif multimask_output and not self.samurai_mode:
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(B, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        if sam_output_tokens.size(1) > 1:
            sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
    else:
        best_iou_inds = 0
        low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

    obj_ptr = self.obj_ptr_proj(sam_output_token)
    if self.pred_obj_scores:
        if self.soft_no_obj_ptr:
            lambda_is_obj_appearing = object_score_logits.sigmoid()
        else:
            lambda_is_obj_appearing = is_obj_appearing.float()
        if self.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

    return (
        low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr,
        object_score_logits, ious[0][best_iou_inds],
        kf_ious[best_iou_inds] if kf_ious is not None else None,
    )
