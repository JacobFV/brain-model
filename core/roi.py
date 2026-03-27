"""Brain region of interest (ROI) utilities using the Destrieux cortical atlas."""

import warnings

import numpy as np

# Map friendly names -> Destrieux atlas label names
ROI_LABEL_MAP = {
    "Visual (V1/V2)":     ["S_calcarine", "G_cuneus"],
    "Occipital":          ["G_occipital_sup", "G_occipital_middle", "Pole_occipital"],
    "Auditory (A1)":      ["G_temp_sup-G_T_transv"],
    "Broca's area":       ["G_front_inf-Opercular", "G_front_inf-Triangul"],
    "Wernicke's area":    ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
    "Fusiform (FFA)":     ["G_oc-temp_lat-fusifor"],
    "Parahipp. (PPA)":    ["G_oc-temp_med-Parahip"],
    "Frontal sup.":       ["G_front_sup"],
    "Angular/TPJ":        ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "Precuneus":          ["G_precuneus"],
    "Motor":              ["G_precentral"],
    "Somatosensory":      ["G_postcentral"],
    "Temporal mid.":      ["G_temporal_middle"],
    "Temporal inf.":      ["G_temporal_inf"],
    "Insula":             ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
    "Cingulate ant.":     ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    "Cingulate post.":    ["G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
    "Orbital frontal":    ["G_orbital", "G_rectus"],
    "Frontal mid.":       ["G_front_middle"],
    "Temporal pole":      ["Pole_temporal"],
}


def get_roi_indices(roi_map: dict[str, list[str]] | None = None) -> dict[str, np.ndarray]:
    """
    Build a mapping from human-readable brain region names to vertex indices
    in fsaverage5 (20484 vertices total: 10242 left + 10242 right).
    """
    from nilearn import datasets

    if roi_map is None:
        roi_map = ROI_LABEL_MAP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = datasets.fetch_atlas_surf_destrieux()

    lh = np.array(atlas["map_left"])   # (10242,)
    rh = np.array(atlas["map_right"])  # (10242,)
    full_map = np.concatenate([lh, rh])  # (20484,)

    labels = [str(l) for l in atlas["labels"]]

    roi_indices = {}
    for friendly_name, atlas_names in roi_map.items():
        indices = []
        for aname in atlas_names:
            if aname in labels:
                label_idx = labels.index(aname)
                indices.append(np.where(full_map == label_idx)[0])
        if indices:
            roi_indices[friendly_name] = np.concatenate(indices)

    return roi_indices


def roi_means(preds: np.ndarray, roi_indices: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """Compute mean activation per ROI from a single-timestep or mean-activation array."""
    if roi_indices is None:
        roi_indices = get_roi_indices()
    act = preds.mean(axis=0) if preds.ndim == 2 else preds
    return {name: float(act[idx].mean()) for name, idx in roi_indices.items()}
