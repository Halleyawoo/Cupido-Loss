# 🫁 Cupido Loss for Structure-Aware Airway and Vessel Segmentation

Official PyTorch implementation of **Cupido Loss**, introduced in our MICCAI Workshop submission.

---

## 🧩 Overview

Cupido Loss is a unified structure-aware loss function designed for vascular and airway segmentation tasks.  
It integrates **Directional Consistency** and **Union Loss** components to ensure accurate, smooth, and continuous vessel predictions across multiple scales.


---

## 🧠 Components

### 1️⃣ Directional Consistency Loss
Encourages alignment between the predicted and ground-truth gradient orientations, ensuring smooth directional transitions in curved and branching vessels.

### 2️⃣ Union Loss
Balances global completeness and local geometric fidelity using skeleton- and distance-based priors to handle vessels of varying diameters.

---

## ⚙️ Pseudocode

Below is a simplified pseudocode description of the **Cupido Loss**:

```python
# Pseudocode for Cupido Loss
def cupido_loss(pred, gt):
    L_direction = direction_loss(pred, gt)
    L_union = union_loss(pred, gt)
    total_loss = α * L_direction + γ * L_union
    return total_loss

## 🔓 Code Availability
The source code and trained models will be released upon paper acceptance.
