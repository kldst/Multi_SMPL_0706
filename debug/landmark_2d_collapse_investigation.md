# Dense-landmark 2D 塌縮成直線 — 問題調查與修復

**日期**:2026-07-05
**模型**:`vggt_multi_0621_mamma_demo_eval_bundle`,dense-landmark head(`vggt/heads/smpl_dense_landmark_head.py`)
**對照**:舊設計 `yian_vggt_smpl/training/smplx_model.py::DenseLandmarkHead`(能訓起來)

---

## 1. 症狀

單幀 overfit(2 人 / 4 view)訓到 `checkpoint_300`,demo 顯示:

- **SMPL mesh(3D + 重投影)正確** —— 貼合人體,重投影 L2 ~40px。
- **mask 正確** —— IoU 0.80 / 0.65,身分分離乾淨。
- **2D landmark 塌成一條垂直線**,而且看起來「沒有朝向」。

---

## 2. 架構前提(先釐清)

- Landmark head 是 **direct 2D per-view**(當初刻意選的,不是 3D gauge + 重投影)。
  → 所以 landmark **本來就不經 gauge space**,也不該經。gauge 只用在 3D SMPL mesh。
- 每個 person query 對每個 view 的 patch tokens 做 cross-attention,直接輸出該視角的正規化 2D。

---

## 3. 調查過程與證據

### 3.1 先量 pred vs GT 的分佈(舊 checkpoint,[-1,1] 慣例)

| view/person | pred 中心 | GT 中心 | pred **x** std | GT **x** std | pred **y** std | GT **y** std |
|---|---|---|---|---|---|---|
| v0 p0 | (0.47,−0.18) | (0.50,−0.19) | **0.020** | 0.089 | 0.336 | 0.393 |
| v0 p1 | (−0.08,−0.18) | (−0.19,−0.19) | **0.018** | 0.064 | 0.302 | 0.347 |
| v1 p0 | (−0.19,−0.12) | (−0.24,−0.04) | **0.011** | 0.070 | 0.259 | 0.215 |
| v1 p1 | (0.33,0.01) | (0.26,0.08) | **0.013** | 0.046 | 0.290 | 0.240 |

**三件事同時成立:**
1. 每個人的 2D **中心正確**(誤差 ~0.1)→ 身分綁定、定位都對。
2. **y(垂直)分佈正確** → 身高方向撐得開。
3. **x(水平)塌掉** → pred x std 只有 GT 的 1/4~1/5 → 512 點擠成一條垂直線。

→ **結論**:不是 gauge、不是視覺化 bug(GT 用同一條 denorm 路徑疊得準;pred 中心/y 都對)。是 x 方向的真實塌縮。

### 3.2 第一次修法(不夠)

對照舊 repo,補了兩個舊有、新版漏掉的設計:

- **座標輸出加 `sigmoid` → [0,1]**(舊版有,新版是 raw `nn.Linear`,無 bound)。
- **GT 正規化改 [0,1]**(對齊 sigmoid)。
- **GNLL 改回舊版 clipped 形式**:`clip(sq/(2σ²), max=25) + 2·logσ`。

重訓 `mamma_overfit_newlandmark/checkpoint_300`,`loss_landmark_l2` 0.21 → 0.03(×518 ≈ 15px),**但 x 仍塌**:

| view/person | pred x std | GT x std | pred y std | GT y std |
|---|---|---|---|---|
| v0 p0 | **0.012** | 0.045 | 0.195 | 0.197 |
| v0 p1 | **0.007** | 0.032 | 0.172 | 0.174 |
| v1 p0 | **0.006** | 0.035 | 0.121 | 0.108 |
| v1 p1 | **0.005** | 0.023 | 0.132 | 0.120 |

pred 範圍 x[0.22,0.70](sigmoid 正常運作),y 完全對,**x 還是窄 3~5×**。

> 附註:在 [0,1] 全圖正規化下,站姿的人 x 展開只有 ~0.04(y ~0.15),`sq` 很小,**clip 25 根本不觸發**(等於沒 clip)。所以 clipped-GNLL 沒起作用。

### 3.3 找到真兇 — query 被 person_token 淹沒

量 query 的兩個組成的量級:

```
landmark_embed  每 token norm = 0.726   (512 個彼此的 per-dim std 只有 0.0227 → 幾乎全相同)
person_token    每 token norm = 54.05
比值 = 74×
```

query 定義為 `landmark_embed[i] + person_token[p]`:
- `person_token[p]`(norm **54**)是 512 個 landmark **共用的巨大常數**。
- 區分「我是第 i 個 vertex」的 `landmark_embed`(norm 0.7,512 個彼此差異 std 0.0227)被淹沒 **74 倍**。

→ 512 個 query 幾乎一模一樣 → attend 相同 patch → 輸出擠成一條線。
→ y 還能撐開,是因為粗結構還漏得出一點訊號;x 這種細結構直接死掉。

**為什麼 `landmark_embed` 這麼弱?** init 是 `torch.randn(...) * 0.02`,太小;加上被 norm-54 的 person_token 主宰,梯度無法讓 512 個 embedding 分化(訓練後也才 std 0.0227)。

**舊 repo 為何沒事**:`self.query = nn.Embedding(512, d_model)`(default init std≈1、norm≈16),而且是**單人 head、不加 person_token**(身分來自 mask-fused patch)→ per-vertex 身分很強、不受污染 → 能解析 x。

---

## 4. 排除的假設

| 假設 | 判定 | 理由 |
|---|---|---|
| 沒轉到 gauge space | ❌ 非原因 | landmark 是 direct 2D,本來就不經 gauge;SMPL 3D 的 gauge 反轉是分開且正確的 |
| 視覺化 denorm 有 bug | ❌ 非原因 | GT 用同一路徑疊得準;pred 中心/y 都對得上 |
| 只訓練一筆資料造成 | ❌ 非原因 | 單幀 overfit 反而讓 x **更容易**擬合;連一幀的 x 都學不出 → 是架構表達不了,非資料量。y 在同一幀擬合得好也證明單幀訓練本身沒問題 |
| **query 被 person_token 淹沒 74×** | ✅ **真因** | 實測量級差 74 倍,512 query 幾乎相同 |

---

## 5. 修復

### 5.1 保留的改動(正確、無害)
- `dec_xy` 加 `sigmoid` → [0,1](`smpl_dense_landmark_head.py`)。
- dataset GT 正規化 → [0,1](`sys_smpl_multi.py`)。
- GNLL 改 clipped 形式(`loss_smpl.py::compute_landmark_loss`)。
- 所有 denorm 視覺化改 [0,1](demo + debug_05/06/07)。

### 5.2 真正的 fix — 重新平衡 query(`smpl_dense_landmark_head.py`)
1. **`landmark_embed` init `*0.02` → std≈1**(`torch.randn(1, L, query_dim)`),給 512 個足夠空間分化。
2. **person_token 先過 `LayerNorm` 再加**(`self.person_ln`):把 norm 54 拉回 ~32,和 landmark_embed 同量級;LN 只改 scale、保留身分方向 → **disentangling 不受影響**。

smoke test 確認:`landmark_embed` per-dim std 0.023 → **0.999**(512 個強烈分化),forward/backward 正常。

---

## 6. 重訓與驗證

```bash
cd /mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle
CUDA_VISIBLE_DEVICES=0 ./run_overfit.sh     # 清掉舊 newlandmark/ckpts 重訓
```

- `mamma_overfit.yaml`:`resume_checkpoint_path` 保持 `ckpt/model.pt`(只載 aggregator;head 全部從零訓);`landmark_loss_type: gnll`;`exp_name: mamma_overfit_newlandmark`(想保留舊壞 ckpt 就改成 `..._newlandmark2`)。

**驗證(關鍵)**:重訓後跑分佈診斷,看 **pred x std 是否追上 GT x std**(目標 ~0.03~0.045,而非 0.005~0.012),demo 目視 **x 鋪開、點蓋滿身體**。

若修好後仍塌 → 再懷疑 loss / 資料;但本次修的是實測證實的直接病灶,信心高。

---

## 附:相關檔案
- Head:`vggt/heads/smpl_dense_landmark_head.py`
- Loss:`training/loss_smpl.py::compute_landmark_loss`
- Dataset GT:`training/data/datasets/sys_smpl_multi.py`(`emit_landmarks`)
- Demo:`demo_gradio_landmark_mask.py` / `run_demo_landmark_mask.sh`
- 舊對照:`/mnt/train-data-4-hdd/yian/yian_vggt_smpl/training/smplx_model.py::DenseLandmarkHead`
