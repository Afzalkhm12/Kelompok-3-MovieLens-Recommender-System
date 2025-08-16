
import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = HERE
FIG_DIR = os.path.join(ART_DIR, 'figs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Model defs (sesuai training) =======
class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.use_bias = bias
        if bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
        else:
            self.user_bias = None
            self.item_bias = None
    def forward(self, u, i):
        ue = self.user_emb(u); ie = self.item_emb(i)
        dot = (ue * ie).sum(dim=1)
        if self.use_bias:
            dot = dot + self.user_bias(u).squeeze(1) + self.item_bias(i).squeeze(1)
        return dot

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, mlp_dims=(256,128,64), dropout=0.3):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        layers = []
        in_dim = emb_dim*2
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)
    def forward(self, u, i):
        ue = self.user_emb(u); ie = self.item_emb(i)
        x = torch.cat([ue, ie], dim=1)
        return self.mlp(x).squeeze(1)

# ======= Load artefak =======
st.set_page_config(page_title="Rekomendasi Film — MovieLens", layout="wide")
st.title("🎬 Sistem Rekomendasi Film — MovieLens (MF vs NCF)")
st.caption("Evaluasi RMSE & MAE, plus rekomendasi Top-K")

idx_info = json.load(open(os.path.join(ART_DIR, 'idx_mappings.json')))
user2idx = {int(k): int(v) for k,v in idx_info['user2idx'].items()}
item2idx = {int(k): int(v) for k,v in idx_info['item2idx'].items()}
R_MIN, R_MAX = idx_info['R_MIN'], idx_info['R_MAX']

movies_small = pd.read_csv(os.path.join(ART_DIR, 'movies_small.csv'))
metrics = pd.read_csv(os.path.join(ART_DIR, 'metrics.csv'))
best_info = json.load(open(os.path.join(ART_DIR, 'best_model.json')))

n_users = len(user2idx); n_items = len(item2idx)

# sidebar — pilih model
models_avail = metrics['model'].dropna().unique().tolist()  # ex: ['MF','NCF']
default_model = best_info.get('best_model', models_avail[0] if models_avail else 'MF')
model_choice = st.sidebar.selectbox("Pilih Model", options=models_avail, index=models_avail.index(default_model) if default_model in models_avail else 0)

# map ke nama checkpoint
ckpt_map = {'MF': 'final_MF.pt', 'NCF': 'final_NCF.pt'}
ckpt_name = ckpt_map.get(model_choice)
ckpt_path = os.path.join(ART_DIR, ckpt_name)
if not os.path.exists(ckpt_path):
    st.error(f"Checkpoint {ckpt_name} tidak ditemukan di {ART_DIR}")
    st.stop()

# rekonstruksi model dari state dict
state = torch.load(ckpt_path, map_location='cpu')
sd = state['model_state_dict']

if model_choice == 'MF':
    emb_dim = sd['user_emb.weight'].shape[1]
    model = MF(n_users, n_items, emb_dim=emb_dim, bias=True)
else:
    emb_dim = sd['user_emb.weight'].shape[1]
    # infer mlp hidden dims (ambil semua 'mlp.*.weight' kecuali output)
    linear_keys = [k for k in sd.keys() if k.startswith('mlp.') and k.endswith('.weight')]
    linear_ids = sorted([int(k.split('.')[1]) for k in linear_keys if k.split('.')[2]=='weight'])
    outs = []
    for lid in linear_ids:
        W = sd[f'mlp.{lid}.weight']
        outs.append(W.shape[0])
    mlp_dims = tuple(outs[:-1]) if outs and outs[-1]==1 else tuple(outs) if outs else (256,128,64)
    model = NCF(n_users, n_items, emb_dim=emb_dim, mlp_dims=mlp_dims, dropout=0.0)

model.load_state_dict(sd); model.eval(); model.to(DEVICE)

# tampilkan metrics
st.subheader("📊 Evaluasi (Test Set)")
st.dataframe(metrics.style.highlight_min(['test_rmse','test_mae'], color='#b6e3ff'), use_container_width=True)

# EDA (jika ada)
st.subheader("🔎 EDA (opsional)")
cols = st.columns(3)
for i, (fn, cap) in enumerate([
    ('rating_distribution.png','Distribusi Rating'),
    ('interaksi_per_user.png','Interaksi per User'),
    ('interaksi_per_item.png','Interaksi per Item'),
]):
    p = os.path.join(FIG_DIR, fn)
    if os.path.exists(p):
        with cols[i%3]:
            st.image(p, caption=cap, use_container_width=True)

# rekomendasi
st.subheader("🎯 Rekomendasi")
all_user_ids = sorted(user2idx.keys())
default_user = all_user_ids[len(all_user_ids)//2] if all_user_ids else (next(iter(user2idx)) if user2idx else 1)
user_raw = st.number_input("Masukkan User ID (raw)", value=default_user, step=1)
topk = st.slider("Top-K", 5, 30, 10, 1)
genre_filter = st.text_input("Filter Genre (opsional, mis. Action|Comedy)", "")

# watched masking
watched_train = {}
wt_path = os.path.join(ART_DIR, 'watched_train.json')
if os.path.exists(wt_path):
    tmp = json.load(open(wt_path))
    watched_train = {int(k): set(v) for k,v in tmp.items()}

inv_item = {v:k for k,v in item2idx.items()}

@torch.no_grad()
def recommend(u_raw, topk=10, genre_filter=None):
    if u_raw not in user2idx:
        return pd.DataFrame(columns=['movieid','title','genres','score'])
    u = user2idx[u_raw]
    seen = watched_train.get(u, set())
    candidates = np.array([it for it in range(n_items) if it not in seen], dtype=np.int64)
    if len(candidates) == 0:
        return pd.DataFrame(columns=['movieid','title','genres','score'])
    u_tensor = torch.tensor([u]*len(candidates), dtype=torch.long, device=DEVICE)
    i_tensor = torch.tensor(candidates, dtype=torch.long, device=DEVICE)
    scores = model(u_tensor, i_tensor).detach().cpu().numpy()
    scores = np.clip(scores, R_MIN, R_MAX)
    cand_movieids = [inv_item[int(i)] for i in candidates]
    rec_df = pd.DataFrame({'movieid': cand_movieids, 'score': scores})
    rec_df = rec_df.merge(movies_small, on='movieid', how='left')
    if genre_filter:
        rec_df = rec_df[rec_df['genres'].fillna('').str.contains(genre_filter, case=False, na=False)]
    return rec_df.sort_values('score', ascending=False).head(topk).reset_index(drop=True)

if st.button("Dapatkan Rekomendasi"):
    t0 = time.time()
    recs = recommend(user_raw, topk=topk, genre_filter=genre_filter.strip() or None)
    st.write(f"Hasil untuk User **{user_raw}** (Top-{topk})")
    st.dataframe(recs, use_container_width=True)
    st.caption(f"Selesai dalam {time.time()-t0:.2f}s")
