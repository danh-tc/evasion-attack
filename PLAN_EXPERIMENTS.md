# Kế hoạch thực nghiệm — RaPA trên Object Detection (COCO)

## Cấu hình

**Dataset:** COCO val2017
- Dev set: `data/manifests/dev_300.json` (300 ảnh, seed=42)
- Held-out: `data/manifests/val_100.json` (100 ảnh — chỉ chạy 1 lần sau khi freeze config)

**Surrogate:** Faster R-CNN ResNet-50-FPN
- Config: `checkpoints/faster-rcnn_r50_fpn_1x_coco.py`
- Checkpoint: `checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`

**Attack hyperparams (mặc định):** ε=8px, 40 iters, step=2px, momentum=0.9

**Metrics:** ASR (object disappearance rate) + ∆AP (COCO AP@[.5:.95] drop)

---

## Target Models (Cross-family)

| Nhóm | Tên | Backbone | Paradigm |
|---|---|---|---|
| **A — Cùng họ (ResNet-50)** | fcos_r50 | ResNet-50 | anchor-free |
| **A — Cùng họ (ResNet-50)** | deformable_detr | ResNet-50 | transformer |
| **B — Gần họ (CNN khác ResNet)** | yolov3_d53 | Darknet-53 | anchor |
| **B — Gần họ (CNN khác ResNet)** | yolox_l | CSPNet | anchor-free |
| **C — Khác họ (Swin ViT)** | mask_rcnn_swin_t | Swin-T | two-stage |
| **C — Khác họ (Swin ViT)** | dino_swin_l | Swin-L | full-transformer |

**Độ khó transfer kỳ vọng:** A (dễ) → B (trung bình) → C (khó)

---

## Thiết kế phương pháp — Object-Aware Temporal RaPA (OA-TRaPA)

**Ý tưởng cốt lõi:** kết hợp loss object-aware theo feature backbone của OSFD (suppress feature quan trọng / amplify feature vùng lân cận) với kỹ thuật random parameter pruning tự-ensemble của RaPA, nhưng thay việc RaPA lấy trung bình gradient qua nhiều mask trong cùng một iteration (S=5–10 lần forward) bằng **một mask mỗi iteration + momentum đóng vai trò ensemble theo thời gian** — mục tiêu là giữ (hoặc cải thiện) khả năng transfer trong khi giảm chi phí iteration/inference, đồng thời kiểm chứng xem sự kết hợp này có phải là một cơ chế thật (giúp transfer tốt hơn ở nhóm cross-family) hay chỉ là một attack mạnh hơn ở white-box.

**Ràng buộc bắt buộc:** trong cùng một loss sample, nhánh benign `F_i(x; M)` và nhánh adversarial `F_i(T(x+δ); M)` phải dùng chung một mask `M`. Nếu hai nhánh dùng mask khác nhau, loss sẽ lẫn giữa feature-distortion do attack và feature-distortion do random pruning, phá vỡ tín hiệu object-aware của OSFD.

**Vì sao masking trên backbone-only quy về việc mask tham số affine của BatchNorm:** loss OSFD chỉ chạy qua `model.backbone(x)` (bỏ qua neck/RPN/head hoàn toàn — đây chính là lý do OSFD rẻ). Với backbone ResNet-50, gần như không có layer `nn.Linear` nào, nên tập target "Linear + Normalization" của DropConnect (RaPA) thu gọn lại chỉ còn **tham số affine của BatchNorm2d (γ, β)** đối với surrogate này. Đây là điểm khởi đầu rõ ràng, không mơ hồ cho Giai đoạn 1.

### Các biến thể phương pháp

| ID | Mô tả |
|---|---|
| A | OSFD baseline (không mask) — control |
| B | OSFD + Bernoulli DropConnect trên BN affine, **S=1** mask/iteration, momentum mang lịch sử |
| C | OSFD + Bernoulli DropConnect trên BN affine, **S=5** mask/iteration (ensemble tường minh, kiểu RaPA gốc — làm upper bound) |
| D | OSFD + **balanced cyclic group mask** (nhóm theo BN-channel, xoay vòng không lặp lại), S=1 |
| E | D + **feature-guided importance bias** (trộn uniform với first-order sensitivity score lấy từ chính loss OSFD, không phải từ loss classifier) |
| F | E + **temporal stage weighting** (EMA phương sai gradient theo stage qua các mask → hạ trọng số stage không ổn định) |

Objective (dạng tổng quát):

```
max_δ  E_{M~q(M), T~T} [ Σ_i w_i · MSE(k · F_i(T(x+δ); M), F_i(x; M)) ]   s.t. ||δ||_∞ ≤ ε
```

Update rule (MI-FGSM):
```
g_t = ∇_δ L(δ_t; M_t, T_t)
v_t = μ·v_{t-1} + g_t / ||g_t||_1
δ_{t+1} = clip_ε(δ_t + α·sign(v_t))
```

---

## Kế hoạch thực nghiệm theo giai đoạn (có gate — không chạy hết các giai đoạn vô điều kiện)

**Probe set (Giai đoạn 1–4):** dùng 3 target đại diện thay vì cả 6 — `fcos_r50` (nhóm A),
`yolox_l` (nhóm B), `mask_rcnn_swin_t` (nhóm C) — trên ~60–80 ảnh lấy từ `dev_300`, 2 seed
mỗi config (vì masking là stochastic, cần biết variance trước khi tin bất kỳ chênh lệch nào).

### Giai đoạn 0 — Hạ tầng (điều kiện tiên quyết, không phải một thí nghiệm)
- Port loss đa-stage của OSFD + augmentation RRB sang mmdet 3.3.0 (stack của repo này khác với
  vendored OSFD gốc dùng mmcv-full 1.7.1 / mmdet 2.28.2 — không thể chạy code vendor as-is).
- Port cơ chế hook DropConnect của RaPA (forward pre-hook/hook hoán đổi weight) thành module riêng.
- Vòng lặp MI-FGSM với mask provider có thể cắm/thay và số mask S cấu hình được.
- Smoke test: 5 ảnh, 5 iterations, xác nhận loss giảm và ASR/∆AP tính được không lỗi.
- **Điều kiện qua giai đoạn:** pipeline chạy hết end-to-end; chưa có kết luận số liệu nào ở bước này.

### Giai đoạn 1 — Masking có giúp gì không, và S=1 có thay được S=5?
- Chạy method **A, B, C** trên probe set với budget mặc định (ε=8px, 40 iters, step=2px,
  momentum=0.9).
- Đo thêm thời gian chạy thực tế mỗi method (để định lượng mức tiết kiệm compute thật của S=1 so với S=5).
- **Tiêu chí pass:**
  1. B hoặc C phải vượt A rõ rệt **đặc biệt ở target nhóm C (cross-family)** — ngưỡng tạm đề xuất
     ≥5 điểm ASR (sẽ tinh chỉnh sau khi thấy variance qua 2 seed). Nếu chỉ tăng ở target cùng họ,
     nghĩa là sự kết hợp này chưa chứng minh được cơ chế transfer, chỉ là attack cục bộ mạnh hơn — dừng ở đây.
  2. B phải nằm trong khoảng ~2–3 điểm ASR so với C — đây là gate cho giả thuyết "momentum ≈
     ensemble theo thời gian". Nếu không đạt, bỏ nhánh single-mask cho các giai đoạn sau (chấp
     nhận chi phí S≥2).
- **Mục tiêu:** quyết định có nên tiếp tục theo đuổi sự kết hợp RaPA+OSFD hay không, và liệu S=1
  có thể dùng cho tất cả các giai đoạn sau hay không (đây chính là con số quyết định mức giảm
  iteration/inference thực tế).

### Giai đoạn 2 — Structured masking có tốt hơn random thuần không? *(chỉ chạy nếu Giai đoạn 1 pass)*
- Chạy method **D**, so với B (dùng lại số liệu Giai đoạn 1, cùng probe set/seed).
- **Mục tiêu:** xác nhận việc tôn trọng tính liên tục không gian của feature map (mask theo nhóm
  BN-channel thay vì Bernoulli độc lập từng scalar) có thực sự giúp ích đo lường được, hay chênh
  lệch chỉ là nhiễu.

### Giai đoạn 3 — Feature-guided importance có phải đóng góp thật không? *(chỉ chạy nếu D > B ở Giai đoạn 2)*
- Chạy method **E**, so với D.
- **Mục tiêu:** đây là phép thử cho claim novelty mạnh nhất (tín hiệu importance lấy từ chính loss
  object-aware OSFD, không phải từ loss classifier như pilot study OBD gốc của RaPA). Nếu E không
  vượt D một cách có ý nghĩa, nên chọn D (ít hyperparameter hơn: không cần tune λ, τ).

### Giai đoạn 4 — Temporal stage weighting có thêm giá trị không? *(chỉ chạy nếu E > D ở Giai đoạn 3)*
- Chạy method **F**, so với E, đặc biệt chú ý vào target nhóm C (cross-family) — đây là giả thuyết
  riêng về stability cho cross-family.
- **Mục tiêu:** quyết định có đưa stage weighting vào method cuối cùng hay không, hoặc để lại như
  một hướng mở rộng trong tương lai nếu tín hiệu yếu.

### Giai đoạn 5 — Chạy full *(chỉ sau khi đã chọn được method thắng)*
- Method thắng + baseline A, trên full `dev_300`, đủ 6 target (cả 2 model mỗi nhóm A/B/C),
  2–3 seed.
- Freeze config, sau đó chạy đúng 1 lần trên `val_100` (held-out) để lấy số báo cáo.

### Ước tính số lượng run

| Giai đoạn | Method mới | Target | Ảnh | Seed | Số run |
|---|---|---|---|---|---|
| 1 | 3 (A,B,C) | 3 | ~70 | 2 | 18 |
| 2 | 1 (D) | 3 | ~70 | 2 | 6 |
| 3 | 1 (E) | 3 | ~70 | 2 | 6 |
| 4 | 1 (F) | 3 | ~70 | 2 | 6 |
| 5 | 2 (winner + A) | 6 | 300 | 2–3 | ~24–36 |
| Val cuối | 2 (winner + A) | 6 | 100 | 1 | 12 |

Tổng ≈ 72–84 run nếu đi hết cả chuỗi; các gate tồn tại chính là để phần lớn các nhánh dừng sớm
thay vì đốt compute cho một giả thuyết đã thất bại từ sớm với chi phí rẻ.

---

## Hạ tầng code (Giai đoạn 0, đã implement)

Package `evasion_attack/` + `scripts/` implement Giai đoạn 0 và method A/B/C (Giai đoạn 1).
Method D/E/F cố tình để trống (`get_method("D")` sẽ raise lỗi) cho đến khi Giai đoạn 1 pass —
đúng nguyên tắc "không đầu tư vào giai đoạn sau khi giai đoạn trước chưa xác nhận".

**Thứ tự chạy trên máy Ubuntu+GPU:**
1. `bash setup_env.sh` — cài môi trường, tải checkpoint, tải COCO val2017, sinh manifest, và tự
   động chạy `scripts/check_env.py` ở cuối (đã sửa: checkpoint phải tải xong trước khi check_env
   chạy, nếu không sẽ lỗi `FileNotFoundError`).
2. Nếu `check_env.py` pass: `python scripts/run_attack.py --n-images 5 --n-iters 5 --out results/smoke.json`
   — smoke test nhỏ, đây là lần đầu tiên nhánh evaluate (inference trên target model, ASR, AP)
   thực sự được chạy (check_env.py chỉ test nhánh attack qua surrogate, chưa test nhánh eval).
3. Giai đoạn 1 probe: `python scripts/run_sweep.py --rates 0.05 --masks 1,5 --n-images 70 --seeds 0,1 --targets fcos_r50,yolox_l,mask_rcnn_swin_t --out results/stage1_probe.json`

**Lưu ý còn tồn tại (chưa test được vì máy dev không có GPU/mmdet cục bộ):** thứ tự kênh màu
RGB/BGR trong `evasion_attack/utils/preprocess.py`, và việc `model.dataset_meta["classes"]` có
tồn tại đúng như kỳ vọng sau `init_detector` hay không (dùng trong `model_zoo.py`/`eval/metrics.py`).
Nếu `run_attack.py`/`run_sweep.py` lỗi ở bước eval, đây là hai chỗ nên kiểm tra đầu tiên.
