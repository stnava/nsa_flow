"""PBMC3k (10x, hg19, 2700 cells): canonical cell-type marker groups among ~1000 variable genes.

Cached by ``nsa-flow-bench fetch pbmc3k --download`` (checksummed 10x tarball -> ``pbmc3k_counts.npz`` +
``pbmc3k_genes.npy`` in the data home).  Preprocessing: log1p of counts per 10,000; features = labelled
markers + the 1000 highest-variance other genes seen in >= 3 cells.  Marker groups are the benchmark authors'
typing of the standard Seurat PBMC3k lists (groups with fewer than 3 detected genes are dropped); review
before citing.
"""
import numpy as np

from .base import Dataset, dataset
from .cache import data_home, download, require, safe_extract, scratch_dir

URL = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
SHA256 = "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5"
MARKERS = {
    "T cell": "CD3D CD3E CD2 IL7R LTB".split(),
    "B cell": "MS4A1 CD79A CD79B CD37 BANK1".split(),
    "NK/cytotoxic": "NKG7 GNLY GZMB PRF1 CTSW GZMA CST7".split(),
    "myeloid": "LYZ S100A8 S100A9 CD14 FCN1 CST3".split(),
    "platelet": "PPBP PF4 GNG11 SDPR".split(),
}


def fetch():
    """Download, verify and convert PBMC3k into the data home (explicit; called by `fetch --download`)."""
    import scipy.io
    import scipy.sparse as sp
    tmp = scratch_dir()
    tar = download(URL, tmp / "pbmc3k.tar.gz", SHA256)
    out = tmp / "x"
    out.mkdir()
    safe_extract(tar, out)
    d = out / "filtered_gene_bc_matrices" / "hg19"
    M = scipy.io.mmread(d / "matrix.mtx").tocsc()
    genes = [line.split("\t")[1] for line in (d / "genes.tsv").read_text().splitlines()]
    home = data_home()
    sp.save_npz(home / "pbmc3k_counts.npz", M)
    np.save(home / "pbmc3k_genes.npy", np.array(genes))


@dataset("pbmc3k", source="10x Genomics PBMC3k (public; fetch --download)", fetch=fetch,
         description="5 cell-type marker groups (27 genes) among 1000 variable genes, 2700 cells")
def load_pbmc3k(n_hvg=1000):
    import scipy.sparse as sp
    C = sp.load_npz(require("pbmc3k_counts.npz", "run `nsa-flow-bench fetch pbmc3k --download`")).tocsc()
    genes = list(np.load(require("pbmc3k_genes.npy"), allow_pickle=True))
    L = C.multiply(1e4 / np.asarray(C.sum(0)).ravel()).tocsc()
    L.data = np.log1p(L.data)
    L = L.T.tocsr()                                              # cells x genes
    seen = np.asarray((C > 0).sum(1)).ravel() >= 3
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
    first = {g: i for i, g in reversed(list(enumerate(genes)))}
    lab = {first[g]: gi for gi, ms in enumerate(MARKERS.values()) for g in ms if g in first and seen[first[g]]}
    counts = np.bincount(list(lab.values()), minlength=len(MARKERS))
    lab = {i: g for i, g in lab.items() if counts[g] >= 3}
    order = [i for i in np.argsort(-var) if seen[i] and i not in lab][:n_hvg]
    cols = list(lab) + order
    remap = {g: n for n, g in enumerate(sorted(set(lab.values())))}
    groups = np.array([remap[lab[c]] for c in lab] + [-1] * len(order))
    return Dataset("pbmc3k", L[:, cols].toarray(), groups=groups, standardize=True, draw_n=1500,
                   meta={"annotation": "canonical markers typed by the authors", "groups": list(MARKERS)})
