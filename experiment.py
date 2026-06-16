"""
Delegated Quantum Inference for Retinal OCT over a Noisy Quantum Link.
Reproduces every table and figure in the paper from a single run.

Usage:
    python experiment.py --data_dir /path/to/RetinalOCT_Dataset
The dataset folder must contain train/ and test/ ImageFolder subdirectories.
GPU is auto-detected (uses 1500/500 subsample); CPU uses 900/300.
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pennylane as qml, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="OCT-C8 folder with train/ and test/")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", DEV)

    IMG, N, L = 224, 6, 2
    SUB_TRAIN, SUB_TEST = (1500, 500) if DEV == "cuda" else (900, 300)
    GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    KINDS = ["depolarizing", "amplitude", "phase"]

    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_full = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=tf)
    test_full = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tf)
    SHARED = train_full.classes; NC = len(SHARED); name2idx = {c: i for i, c in enumerate(SHARED)}
    test_name = {i: c for c, i in test_full.class_to_idx.items()}
    keep = [k for k, (_, lab) in enumerate(test_full.samples) if test_name[lab] in name2idx]

    class Remap(torch.utils.data.Dataset):
        def __init__(s, b, idx): s.b, s.idx = b, idx
        def __len__(s): return len(s.idx)
        def __getitem__(s, i):
            x, lab = s.b[s.idx[i]]; return x, name2idx[test_name[lab]]
    test_ds = Remap(test_full, keep)
    print(f"classes={SHARED} | train {len(train_full)} | test {len(test_ds)} | subsample {SUB_TRAIN}/{SUB_TEST}")

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = backbone.fc.in_features; backbone.fc = nn.Identity(); backbone = backbone.to(DEV).eval()
    for p in backbone.parameters(): p.requires_grad = False

    def extract(ds, limit):
        idx = np.random.permutation(len(ds))[:limit]
        dl = DataLoader(Subset(ds, idx.tolist()), batch_size=64); Fs, Ys = [], []
        with torch.no_grad():
            for xb, yb in dl: Fs.append(backbone(xb.to(DEV)).cpu()); Ys.append(yb)
        return torch.cat(Fs), torch.cat(Ys)
    print("extracting features...")
    Ftr, Ytr = extract(train_full, SUB_TRAIN); Fte, Yte = extract(test_ds, SUB_TEST)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
    pca = PCA(N).fit(Ftr.numpy())
    Ztr = torch.tensor(pca.transform(Ftr.numpy()), dtype=torch.float)
    Zte = torch.tensor(pca.transform(Fte.numpy()), dtype=torch.float)
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-6; Ztr = (Ztr - zmu) / zsd; Zte = (Zte - zmu) / zsd

    cdev = qml.device("default.qubit", wires=N); mdev = qml.device("default.mixed", wires=N)
    WS = qml.StronglyEntanglingLayers.shape(n_layers=L, n_wires=N)

    def _body(x, w, p, kind):
        qml.AngleEmbedding(x, wires=range(N))
        if p > 0:
            for wi in range(N):
                if kind == "depolarizing": qml.DepolarizingChannel(p, wires=wi)
                elif kind == "amplitude": qml.AmplitudeDamping(p, wires=wi)
                else: qml.PhaseDamping(p, wires=wi)
        qml.StronglyEntanglingLayers(w, wires=range(N))
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    clean_circ = qml.QNode(lambda x, w: _body(x, w, 0.0, "depolarizing"), cdev, interface="torch", diff_method="backprop")
    noisy_circ = qml.QNode(lambda x, w, p, kind: _body(x, w, p, kind), mdev, interface="torch", diff_method="backprop")

    class QHead(nn.Module):
        def __init__(s):
            super().__init__(); s.scale = nn.Parameter(torch.ones(N))
            s.w = nn.Parameter(0.1 * torch.randn(*WS)); s.head = nn.Linear(N, NC)
        def forward(s, z, p=0.0, kind="depolarizing"):
            a = torch.tanh(z * s.scale) * np.pi
            out = clean_circ(a, s.w) if p == 0 else noisy_circ(a, s.w, p, kind)
            return s.head(torch.stack(out, dim=1).float())

    def fit_qhead(seed=0):
        torch.manual_seed(seed); m = QHead(); o = torch.optim.Adam(m.parameters(), lr=0.05); lf = nn.CrossEntropyLoss()
        for _ in range(20):
            perm = torch.randperm(len(Ztr))
            for i in range(0, len(Ztr), 128):
                idx = perm[i:i + 128]; o.zero_grad(); lf(m(Ztr[idx]), Ytr[idx]).backward(); o.step()
        return m.eval()
    qh = fit_qhead(args.seed)

    clf = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, NC))
    co = torch.optim.Adam(clf.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    for _ in range(15):
        perm = torch.randperm(len(Ftr))
        for i in range(0, len(Ftr), 64):
            idx = perm[i:i + 64]; co.zero_grad(); lf(clf(Ftr[idx]), Ytr[idx]).backward(); co.step()

    def probs(p=0.0, k="depolarizing"):
        o = []
        with torch.no_grad():
            for i in range(0, len(Zte), 256): o.append(torch.softmax(qh(Zte[i:i + 256], p, k), 1))
        return torch.cat(o)
    def logits(p=0.0, k="depolarizing"):
        o = []
        with torch.no_grad():
            for i in range(0, len(Zte), 256): o.append(qh(Zte[i:i + 256], p, k))
        return torch.cat(o)
    def ECE(P, y, bins=10):
        c = P.max(1).values.numpy(); pr = P.argmax(1).numpy(); cr = (pr == y.numpy()).astype(float); e = 0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins; m = (c > lo) & (c <= hi)
            if m.sum() > 0: e += abs(cr[m].mean() - c[m].mean()) * m.mean()
        return float(e)
    Brier = lambda P, y: float(((P - torch.eye(NC)[y]) ** 2).sum(1).mean())
    NLL = lambda P, y: float(F.nll_loss(torch.log(P + 1e-9), y))

    print("\n===== TABLE I (clean per-class, quantum head) =====")
    print(classification_report(Yte.numpy(), probs(0).argmax(1).numpy(), target_names=SHARED, digits=2))
    print("classical reference acc:", round((clf(Fte).argmax(1) == Yte).float().mean().item(), 3))

    M = {k: {"acc": [], "ece": []} for k in KINDS}
    for k in KINDS:
        for p in GRID:
            P = probs(p, k); M[k]["acc"].append((P.argmax(1) == Yte).float().mean().item()); M[k]["ece"].append(ECE(P, Yte))
    print("\n===== TABLE II (accuracy & ECE vs p) =====")
    for j, p in enumerate(GRID):
        print(f"p={p:<5} " + " ".join([f"{k[:4]} acc={M[k]['acc'][j]:.3f} ece={M[k]['ece'][j]:.3f}" for k in KINDS]))
    plt.figure(figsize=(5, 3.4))
    for k, mk in zip(KINDS, 'os^'): plt.plot(GRID, M[k]['acc'], mk + '-', label=k)
    plt.axhline((clf(Fte).argmax(1) == Yte).float().mean().item(), ls='--', c='gray', label='classical')
    plt.xlabel('link noise p'); plt.ylabel('accuracy'); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'fig_netsweep.png'), dpi=200); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for k, mk in zip(KINDS, 'os^'):
        ax[0].plot(GRID, M[k]['acc'], mk + '-', label=k); ax[1].plot(GRID, M[k]['ece'], mk + '-', label=k)
    ax[0].set_title('Test accuracy'); ax[0].set_xlabel('link noise p'); ax[0].set_ylabel('accuracy'); ax[0].legend(fontsize=7)
    ax[1].set_title('Expected calibration error'); ax[1].set_xlabel('link noise p'); ax[1].set_ylabel('ECE'); ax[1].axhline(0.10, ls=':', c='r'); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'fig_reliability.png'), dpi=200); plt.close()

    print("\n===== TABLE III (link budget, first-crossing; acc>=0.85, ECE<=0.10) =====")
    def first_cross(vals, ok):
        p = 0.0
        for i, g in enumerate(GRID):
            if ok(vals[i]): p = g
            else: break
        return p
    for k in KINDS:
        pa = first_cross(M[k]['acc'], lambda v: v >= 0.85); pe = first_cross(M[k]['ece'], lambda v: v <= 0.10)
        print(f"  {k:<13} acc>=0.85: {pa:.2f}   ECE<=0.10: {pe:.2f}")

    print("\n===== TABLE IV (extended calibration, depolarizing) =====")
    for p in [0.0, 0.10, 0.20, 0.30]:
        P = probs(p); print(f"p={p:<4} ECE={ECE(P, Yte):.3f} Brier={Brier(P, Yte):.3f} NLL={NLL(P, Yte):.3f}")
    print("ECE vs bins {5,10,15,20} (clean):", [round(ECE(probs(0.0), Yte, b), 3) for b in [5, 10, 15, 20]])
    accs = [(fit_qhead(s)(Zte).argmax(1) == Yte).float().mean().item() for s in range(5)]
    print(f"multi-seed acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    P0 = probs(0.0); n = len(Yte)
    ba = [(P0[idx].argmax(1) == Yte[idx]).float().mean().item() for idx in [np.random.randint(0, n, n) for _ in range(500)]]
    print(f"acc 95% CI: [{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]")
    h = len(Yte) // 2; Lc = logits(0.0); T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    def _cl(): opt.zero_grad(); l = F.cross_entropy(Lc[:h] / T, Yte[:h]); l.backward(); return l
    opt.step(_cl); T = float(T.detach())
    for p in [0.0, 0.20]:
        Lp = logits(p)[h:]; yy = Yte[h:]
        print(f"temp T={T:.2f} p={p}: ECE {ECE(torch.softmax(Lp,1),yy):.3f} -> {ECE(torch.softmax(Lp/T,1),yy):.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    for a, p in zip(ax, [0.0, 0.20]):
        P = probs(p); cf = P.max(1).values.numpy(); pr = P.argmax(1).numpy(); cr = (pr == Yte.numpy()).astype(float); xs = []; ys = []
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10; m = (cf > lo) & (cf <= hi)
            if m.sum() > 0: xs.append(cf[m].mean()); ys.append(cr[m].mean())
        a.plot([0, 1], [0, 1], 'k--'); a.plot(xs, ys, 'o-'); a.set_title(f"reliability (p={p})")
        a.set_xlabel("confidence"); a.set_ylabel("accuracy"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'fig_reliability_diagram.png'), dpi=200); plt.close()

    print("\n===== TABLE V (few-shot) =====")
    def tr_c(X, y, ep=60):
        m = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, NC)); o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3); l2 = nn.CrossEntropyLoss()
        for _ in range(ep): o.zero_grad(); l2(m(X), y).backward(); o.step()
        with torch.no_grad(): return (m(Fte).argmax(1) == Yte).float().mean().item()
    def tr_q(X, y, ep=60):
        m = QHead(); o = torch.optim.Adam(m.parameters(), lr=0.05); l2 = nn.CrossEntropyLoss()
        for _ in range(ep): o.zero_grad(); l2(m(X), y).backward(); o.step()
        with torch.no_grad(): return (m(Zte).argmax(1) == Yte).float().mean().item()
    SHOTS = [5, 10, 20, 50]; fs = {"c": {}, "q": {}}
    for kk in SHOTS:
        cc = []; qq = []
        for s in range(3):
            g = torch.Generator().manual_seed(s); idx = []
            for c in range(NC):
                ci = (Ytr == c).nonzero().flatten(); idx.append(ci[torch.randperm(len(ci), generator=g)[:kk]])
            idx = torch.cat(idx); cc.append(tr_c(Ftr[idx], Ytr[idx])); qq.append(tr_q(Ztr[idx], Ytr[idx]))
        fs["c"][kk] = (np.mean(cc), np.std(cc)); fs["q"][kk] = (np.mean(qq), np.std(qq))
        print(f"  {kk}/class  classical {np.mean(cc):.3f}+/-{np.std(cc):.3f}   quantum {np.mean(qq):.3f}+/-{np.std(qq):.3f}")
    plt.figure(figsize=(5, 3.4))
    plt.errorbar(SHOTS, [fs['c'][k][0] for k in SHOTS], yerr=[fs['c'][k][1] for k in SHOTS], marker='s', capsize=3, label='classical')
    plt.errorbar(SHOTS, [fs['q'][k][0] for k in SHOTS], yerr=[fs['q'][k][1] for k in SHOTS], marker='o', capsize=3, label='quantum')
    plt.xlabel('samples / class'); plt.ylabel('accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'fig_fewshot.png'), dpi=200); plt.close()

    pmean = torch.tensor(pca.mean_, dtype=torch.float); pcomp = torch.tensor(pca.components_, dtype=torch.float)
    acts = {}; grads = {}
    h1 = backbone.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = backbone.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    def f2q(f):
        z = (f - mu) / sd; z = (z - pmean) @ pcomp.t(); z = (z - zmu) / zsd
        return qh.head(torch.stack(clean_circ(torch.tanh(z * qh.scale) * np.pi, qh.w), dim=1).float())
    def f2c(f): return clf((f - mu) / sd)
    def gcam(x, which):
        backbone.zero_grad(); xb = x.unsqueeze(0).to(DEV).requires_grad_(True); feat = backbone(xb).to("cpu")
        lo = f2q(feat) if which == "quantum" else f2c(feat); c = int(lo.argmax()); lo[0, c].backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True); cam = torch.relu((w * acts["v"]).sum(1)).squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    def hconf(x, which, tgt):
        with torch.no_grad():
            f = backbone(x.unsqueeze(0).to(DEV)).to("cpu"); return torch.softmax(f2q(f) if which == "quantum" else f2c(f), 1)[0, tgt].item()
    def cof(x, which):
        with torch.no_grad():
            f = backbone(x.unsqueeze(0).to(DEV)).to("cpu"); return torch.softmax(f2q(f) if which == "quantum" else f2c(f), 1).max().item()
    def delins(x, cam, which, steps=8):
        H, W = x.shape[-2:]; up = torch.tensor(np.kron(cam, np.ones((H // cam.shape[0], W // cam.shape[1]))), dtype=torch.float)
        order = torch.argsort(up.flatten(), descending=True)
        with torch.no_grad():
            f0 = backbone(x.unsqueeze(0).to(DEV)).to("cpu"); tgt = int((f2q(f0) if which == "quantum" else f2c(f0)).argmax())
        per = len(order) // steps; xd = x.clone().view(1, -1); dc = [hconf(x, which, tgt)]
        for s in range(steps): xd[0, order[s * per:(s + 1) * per]] = 0; dc.append(hconf(xd.view(x.shape), which, tgt))
        xi = torch.zeros_like(x).view(1, -1); base = x.view(1, -1); ic = [hconf(torch.zeros_like(x), which, tgt)]
        for s in range(steps): xi[0, order[s * per:(s + 1) * per]] = base[0, order[s * per:(s + 1) * per]]; ic.append(hconf(xi.view(x.shape), which, tgt))
        a = lambda c: float(np.mean([(c[i] + c[i + 1]) / 2 for i in range(len(c) - 1)])); return a(dc), a(ic)
    INV = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])
    picks = {}
    for kk in range(len(test_ds)):
        x, y = test_ds[kk]
        if y not in picks: picks[y] = x
        if len(picks) == NC: break
    fig, ax = plt.subplots(3, NC, figsize=(4 * NC, 11))
    for j, (y, x) in enumerate(sorted(picks.items())):
        im = INV(x).permute(1, 2, 0).clamp(0, 1).numpy(); cc = gcam(x, "classical"); cq = gcam(x, "quantum")
        ax[0, j].imshow(im); ax[0, j].set_title(SHARED[y]); ax[0, j].axis("off")
        ax[1, j].imshow(im); ax[1, j].imshow(np.kron(cc, np.ones((32, 32))), cmap="jet", alpha=0.45); ax[1, j].axis("off")
        ax[2, j].imshow(im); ax[2, j].imshow(np.kron(cq, np.ones((32, 32))), cmap="jet", alpha=0.45); ax[2, j].axis("off")
    plt.suptitle("Grad-CAM: classical (middle) vs quantum (bottom)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fig_gradcam_compare.png"), dpi=200); plt.close()
    NIMG = 40; sample = [test_ds[i] for i in np.random.permutation(len(test_ds))[:NIMG]]
    agg = {"quantum": {"d": [], "i": [], "c": []}, "classical": {"d": [], "i": [], "c": []}}
    for x, y in sample:
        for which in ["quantum", "classical"]:
            cam = gcam(x, which); d, i = delins(x, cam, which)
            agg[which]["d"].append(d); agg[which]["i"].append(i); agg[which]["c"].append(cof(x, which))
    print("\n===== TABLE VI (faithfulness, N=40, mean[95% CI]) =====")
    for which in ["classical", "quantum"]:
        d = np.array(agg[which]["d"]); i = np.array(agg[which]["i"])
        dl, dh = np.percentile([d[np.random.randint(0, len(d), len(d))].mean() for _ in range(500)], [2.5, 97.5])
        il, ih = np.percentile([i[np.random.randint(0, len(i), len(i))].mean() for _ in range(500)], [2.5, 97.5])
        print(f"  {which:<10} deletion {d.mean():.3f}[{dl:.3f},{dh:.3f}]  insertion {i.mean():.3f}[{il:.3f},{ih:.3f}]  mean-conf {np.mean(agg[which]['c']):.3f}")
    h1.remove(); h2.remove()

    print("\n===== baselines on identical PCA-6 features =====")
    lr = LogisticRegression(max_iter=2000).fit(Ztr.numpy(), Ytr.numpy()); print("  logistic regression:", round((lr.predict(Zte.numpy()) == Yte.numpy()).mean(), 3))
    sv = LinearSVC().fit(Ztr.numpy(), Ytr.numpy()); print("  linear SVM         :", round((sv.predict(Zte.numpy()) == Yte.numpy()).mean(), 3))
    mlp = nn.Sequential(nn.Linear(N, 8), nn.ReLU(), nn.Linear(8, NC)); om = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(150): om.zero_grad(); lf(mlp(Ztr), Ytr).backward(); om.step()
    with torch.no_grad(): print(f"  matched MLP        : {(mlp(Zte).argmax(1) == Yte).float().mean().item():.3f}")
    print("\nDONE. Figures saved to", args.out_dir)

if __name__ == "__main__":
    main()
