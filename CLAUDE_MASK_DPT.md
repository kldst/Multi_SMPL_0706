# CLAUDE_MASK_DPT.md — Pixel-level mask head + Hungarian mask cost 改動紀錄

日期：2026-07-10。對應 config：`training/config/mamma_mask_dpt.yaml`。
主文件 `CLAUDE.md` 有摘要，本檔是完整的動機、實作與驗證紀錄。

## 問題背景（為什麼做這次改動）

多人 SMPL（6 個 person query + Hungarian matching）在**人物有接觸**時，
model 會 focus 到別人身上、預測不準。根因分析：

1. **Hungarian matching 在接觸場景失效**：舊 cost 只有
   pose(1.0)/beta(0.1)/mesh_translate(0.3)/presence(0.05)。兩人接觸時
   mesh_translate 幾乎相同、pose（local rotation）也相似 → cost 矩陣兩列打平
   → 配對每個 step 在 A/B 之間翻轉 → slot 被來回拔河，學到「站中間取平均」。
2. **舊的輔助監督（landmark + 37×37 mask）收斂太慢**：這些 head 的輸出只在
   配對「之後」被 gather，沒參與配對本身 —— 病灶（配對翻轉）沒治。
   且 37×37 patch grid 上 ~3.75% 的像素是邊界模糊地帶，分不開接觸的人。

本次 ablation 策略：**landmark 全關**，只上 pixel-level mask head + mask
matching cost，先驗證 mask 是否足以解 identity mixing，之後再考慮加回 landmark。

## 改動清單（8 項）

### 1. 向量化 Hungarian cost 矩陣 + 加入 mask cost — `training/smpl_matching.py`

- `apply_hungarian_matching` 的 P_pred×P_gt 雙層 Python 迴圈改成 broadcasting
  一次算完整張 cost 矩陣，且整段包在 `torch.no_grad()`（配對是硬決策，本來
  就不需要 graph；舊版還會建無用的 autograd graph）。
- 新參數 `cost_mask_weight`（config: `hungarian_cost_mask_weight`）與
  `mask_cost_grid`（config: `hungarian_mask_cost_grid`，預設 32）：
  - pred `person_mask_logits (B,S,P,H,W)` 過 sigmoid、GT `person_mask
    (B,S,P,h,w)` 各自 `adaptive_avg_pool2d` 到 `grid×grid`（**解析度無關**，
    37×37 或 518×518 的輸入都吃）。
  - 每對 (pred i, GT j) 的 soft BCE 用兩個 matmul 算完：
    `-(log(p) @ g.T + log(1-p) @ (1-g).T) / N`。
  - 這是解接觸場景的關鍵：pose/mesh_trans cost 打平時，**2D image mask 仍然
    分得開**（且 4 個 view 一起平均，多視角證據更強）。
- 配對後 matched-pair 的 `hungarian_mask_cost` / `hungarian_presence_cost`
  現在回傳進 loss dict（舊版 `matching_cost_metrics` 算了但沒接出去），
  wandb/tensorboard 可監控配對品質。
- 權重 threading：`loss_smpl.py` 的 `compute_smpl_loss` 新增
  `hungarian_cost_mask_weight` / `hungarian_mask_cost_grid` 參數
  （`loss.smpl` config block 直接 `**self.smpl` 灌入，加 key 即生效）。

### 2. DPT-style pixel-level mask head — `vggt/heads/person_mask_head.py`

新增 `PersonMaskDPTHead`（舊的 37×37 `PersonMaskHead` 保留不動）：

- **DPT trunk 只跑一次**（`DPTHead(feature_only=True)`，吃 aggregator 的
  4 個 intermediate layer [4,11,17,23]）→ per-pixel embedding map
  `(B,S,features,H/down_ratio,W/down_ratio)`。
- person token 過 LayerNorm+Linear 投影到 `embed_dim`(128)，與 1×1 conv 投影
  後的 pixel embedding 做 einsum 內積（Mask2Former-style）
  → `person_mask_logits (B,S,P,H',W')`。P 個人**不會**跑 P 次 DPT。
- 有 learnable log-scale temperature。~33M 參數。
- 記憶體重點：DPT 的 fusion 卷積全部在 ≤148×148 完成，`down_ratio` 只影響
  最後的 bilinear 上採樣 → 全解析度（518）比 259 只多 ~1.6GB。

### 3. 模型接線 — `vggt/models/vggt.py`

新 config knobs（`model` block）：

| knob | 值 | 意義 |
|---|---|---|
| `person_mask_head_type` | `"dot"`（預設）/ `"dpt"` | 舊 37×37 頭 / 新 pixel-level 頭 |
| `person_mask_down_ratio` | 1 | logits 解析度 = 518/ratio（1=518², 2=259²） |
| `person_mask_embed_dim` | 128 | dot-product embedding dim |

forward 裡 dpt 分支把 `aggregated_tokens_list + images + patch_start_idx`
傳給新 head（dot 分支維持原樣）。注意：aux heads 跑在
`autocast(enabled=False)` 區塊內（fp32），與其他 head 一致。

### 4. Dataset 輸出 pixel-level mask GT — `training/data/datasets/sys_smpl_multi.py`

- 新參數 `person_mask_stride`（預設 `None` = 舊行為 patch grid 37×37）：
  GT mask grid = `H_final // stride`。`stride=1` → 518×518。
- 沿用 `rasterize_person_patch_mask`（INTER_AREA soft occupancy）；
  `stride=1` 時 resize 是 no-op，**GT 保持二值** → `loss_mask` floor ≈ 0、
  `mask_soft_iou` 天花板 ≈ 1（見下方實測）。
- **必須與 `model.person_mask_down_ratio` 一致**（stride 1 ↔ ratio 1）。

### 5. Loss 端解析度對齊 — `training/loss_mask.py`

`compute_mask_loss` 遇到 pred/GT 空間解析度不合時，改為把 logits bilinear
resample 到 GT grid（原本直接 raise）。batch/view 維不合仍會 raise。

### 6. 新 config — `training/config/mamma_mask_dpt.yaml`

`mamma_full.yaml` 的 ablation 版（原檔未動，方便 A/B）：

- landmark/contact 全關：`enable_smpl_dense_landmark: False`、
  `emit_landmarks/emit_contact: False`、`weight_landmark/weight_landmark_vis/
  weight_contact/weight_floor_contact: 0`。
- mask：`person_mask_head_type: dpt` + `person_mask_down_ratio: 1` +
  dataset `person_mask_stride: 1` + `weight_mask: 1.0`。
- matching：`hungarian_cost_mask_weight: 1.0`、`hungarian_mask_cost_grid: 32`。
- resume 從已收斂的 SMPL-only checkpoint
  （`training/logs/0621_mamma/ckpts/checkpoint_step_10000.pt`）warm-start，
  避免新 head 的早期垃圾梯度透過共用的 person_tokens 拖垮 SMPL head。
- logging keys 換成 mask 相關（`loss_mask`/`mask_soft_iou`/
  `hungarian_mask_cost`/`hungarian_presence_cost`）。

### 7. Depth→mask trunk warm-start — `training/trainer.py`

`checkpoint.init_mask_trunk_from_depth`（`_load_resuming_checkpoint` 內）：

- `True`：從 **resume checkpoint 本身**拿 `depth_head.*` 複製到
  `person_mask_head.trunk.*`。
- **路徑字串**：從指定 checkpoint 載（本 config 用這個，因為
  `checkpoint_step_10000.pt` 是 `enable_depth: False` 練的、**沒有**
  depth 權重；base VGGT `ckpt/model.pt` 才有）。
- 只在 shape 相同且 resume dict 沒有該 key 時複製；`feature_only` 的
  `output_conv1` shape 不同會被跳過（正常，56 個 tensor 可複製、2 個跳過）。
- 複製 0 個 tensor 會印 **warning**（不再靜默失敗）。
- 動機：depth 邊界 ≈ 人體邊界，trunk 從 depth 權重出發比從零練快很多。
- **啟動時務必檢查 log**：`init_mask_trunk_from_depth: copied 56
  depth_head.* tensors from .../ckpt/model.pt` 代表成功。

### 8. Smoke test 結果（scratchpad `test_mask_dpt_changes.py` 等，session 暫存）

全部 PASS：

- **matching 等價性**：向量化版與舊雙迴圈 reference 在隨機資料上產生完全相同
  的 assignment（pose/beta/mesh_trans/presence 全開）。
- **接觸情境**：兩人 pose/beta/mesh_translate 完全相同（參數 cost 全打平），
  只有 mask cost 能分 → 正確把 slot 配到 mask 重疊的人（需交換 slot 的 case
  也對）。
- **混合解析度**：pred 259² vs GT 37² 的 mask cost / loss 都正常。
- **PersonMaskDPTHead** forward/backward shape 正確，梯度流回 person_tokens。
- **GT-as-pred sanity**：近完美 logits → loss≈0、soft IoU≈1。
- **GPU 記憶體實測**（4070 Ti SUPER 16GB，B=1、4 views @518、frozen
  aggregator、bf16 autocast）：

  | down_ratio | logits | fwd+bwd peak |
  |---|---|---|
  | 2 | 259×259 | 7.95 GiB |
  | **1（採用）** | **518×518** | **9.58 GiB** |

- **loss floor 實測**（真實 `0000.mask.jpg`，GT-as-pred 餵回）：

  | GT grid | soft 像素比例 | BCE floor | soft IoU 上限 |
  |---|---|---|---|
  | 37×37 | 3.75% | 0.0168 | 0.675 |
  | 259×259 | 0.31% | 0.0020 | 0.951 |
  | 518×518（stride 1，二值） | ~0% | ~0 | ~1.0 |

  （soft occupancy 的 BCE 下限 = GT 的 binary entropy，不是 0；stride 1
  下 GT 二值所以回到 0。全零預測 baseline = ln2 ≈ 0.693。）
- 附帶確認：`.mask.jpg` 像素值乾淨（{0,1,2,3}，0=背景、其餘=person_idx+1），
  JPEG 壓縮沒產生雜值，`mask == person_value` 等值判斷安全。

## 怎麼跑

```bash
conda activate mamma
export PYTHONPATH=/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle:/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle/training
cd /mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle/training
torchrun --standalone --nproc_per_node=1 launch.py --config mamma_mask_dpt
```

煙測：config 頂部 `debug_max_sequences: 3` / `debug_max_frames_per_sequence: 2`
+ `limit_train_batches: 20`；正式跑改回 `null`。

## 監控指南（wandb / tensorboard）

- `hungarian_mask_cost`：matched pair 的 mask BCE cost。應隨 mask 變準持續
  下降；它降代表配對越來越被 2D 證據鎖定。
- `loss_mask`：從 ~0.69（全零 baseline）往 ~0 收。
- `mask_soft_iou`：stride 1 下天花板 ~1.0。
- 若 mask 收斂了但接觸 frame 的 `loss_smpl_joints2d` 仍差 → 下一步再開
  landmark head（屆時 mask 已可靠，可當 landmark 的 attention prompt，
  比兩者一起從零練好收斂得多）。

## 已知注意事項 / 坑

1. `person_mask_stride`（dataset）必須跟 `person_mask_down_ratio`（model）
   一致；不一致時 loss 端會自動 resample 但監督品質打折。
2. `init_mask_trunk_from_depth` 指向的 checkpoint 必須真的有 `depth_head.*`
   （SMPL-only checkpoint 沒有）；看啟動 log 的 copied 數量確認。
3. Hungarian mask cost 在訓練最初期（mask 還是垃圾時）約為常數 ~0.69、
   不具判別力，配對暫由 pose/mesh_trans 主導 —— 正常，不用調。
4. stride 1 的 GT 是 `(S,P,518,518)` float32 ≈ 25.7MB/sample，`num_workers: 0`
   下沒問題；若之後開多 worker 注意 shared memory。
5. `accum_steps` 仍是 1（batch=1/step，Hungarian 很吵）；如收斂不穩，優先
   試 4–8。
6. 舊 demo/debug 腳本（`demo_gradio_landmark_mask.py`、debug_04/05/06）假設
   37×37 patch-grid mask；用 dpt head 的 checkpoint 跑它們時視覺化需自行調整
   （loss 端因第 5 項改動不會炸）。

## 尚未做（后续可选）

- landmark head 加回（等 mask ablation 結論）。
- masked cross-attention（Mask2Former-style hard attention bias）於 landmark
  decoder。
- 解凍 aggregator 後段 blocks / LoRA（若 frozen 特徵成為天花板）。
- contact frame oversample / loss 加權。
