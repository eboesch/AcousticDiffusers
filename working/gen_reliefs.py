import numpy as np
import scipy.sparse.linalg
import tqdm
import glob, pathlib

import working.helpers as H
import working.mesher as mesher
import torch
import cv2

# ---------------------------

# args = parse_args()
# INFILE = "tmp/composite_output.png"
# OUTFILE = "rafael.png"
WIDTH = HEIGHT = 256
OFS = 0
GC_ALPHA = 10

# ---------------------------

# load depth image
def gen_hmap(in_file):
    if isinstance(in_file, str):
        hf = H.read_image(in_file, w=WIDTH, h=HEIGHT, format="L")
    else:
        hf = in_file
    # print(np.min(hf), np.max(hf))

    # we expect depth to increase with distance.  Flip it around.
    # hf = (255.0-hf)/255.0

    bkg_h = np.min( hf )

    ROWS, COLS = hf.shape

    # boundaries
    hf[0,:]=0.0
    hf[ROWS-1,:]=0.0
    hf[:,0]=0.0
    hf[:,COLS-1]=0.0

    def grad_compress( gx, gy, alpha=10.0 ):
        mags = np.sqrt( gx*gx + gy*gy + 0.0001 )
        cmags = (1.0/alpha) * np.log( 1.0 + alpha*mags )
        gx = gx / cmags
        gy = gy / cmags
        return gx, gy, cmags

    def rc_to_linind( r,c ):
        return r*COLS+c

    gx = np.zeros((ROWS,COLS),dtype=np.float32)
    gy = np.zeros((ROWS,COLS),dtype=np.float32)
    for row_ind in range(1,ROWS-1):
        for col_ind in range(1,COLS-1):
            gx[row_ind,col_ind] = hf[row_ind,col_ind+OFS]-hf[row_ind,col_ind-1]
            gy[row_ind,col_ind] = hf[row_ind+OFS,col_ind]-hf[row_ind-1,col_ind]        

    comp_gx, comp_gy, cmags = grad_compress( gx, gy, GC_ALPHA)

    row_inds = []
    col_inds = []
    data = []
    b = []
    next_ind = ROWS*COLS
    for row_ind in range(ROWS):
        for col_ind in range(COLS):
            xlc   = rc_to_linind( row_ind, col_ind )
            # boundary condition
            if row_ind == 0 or row_ind == ROWS-1 or col_ind == 0 or col_ind == COLS-1: # or hf[row_ind,col_ind] == bkg_h:
                row_inds.append(xlc); col_inds.append(xlc); data.append(1);
                b.append( 0 )
                continue
            
            # dx constraint
            xlc_l = rc_to_linind( row_ind, col_ind-1 )
            xlc_r = rc_to_linind( row_ind, col_ind+OFS )
            row_inds.append(xlc); col_inds.append(xlc_l); data.append(-1);
            row_inds.append(xlc); col_inds.append(xlc_r); data.append(1);
            # target value
            b.append( comp_gx[row_ind,col_ind] )

    next_ind = ROWS*COLS
    for row_ind in range(ROWS):
        for col_ind in range(COLS):
            xlc   = rc_to_linind( row_ind, col_ind )
            # boundary condition
            if row_ind == 0 or row_ind == ROWS-1 or col_ind == 0 or col_ind == COLS-1: # or hf[row_ind,col_ind] == bkg_h:
                # already got these above
                continue
            
            # dy constraint
            xlc_u = rc_to_linind( row_ind-1, col_ind )
            xlc_d = rc_to_linind( row_ind+OFS, col_ind )
            row_inds.append(next_ind); col_inds.append(xlc_u); data.append(-1);  
            row_inds.append(next_ind); col_inds.append(xlc_d); data.append(1);
            # target value
            b.append( comp_gy[row_ind,col_ind] )
            
            next_ind += 1

    A = scipy.sparse.csc_matrix( (data,(row_inds,col_inds)) )
    b = np.atleast_2d( b ).T

    results = scipy.sparse.linalg.lsqr( A, b, show=False, atol=1e-15, btol=1e-15, conlim=100 )

    new_hf = np.reshape( np.atleast_2d( results[0] ).T, (ROWS,COLS) )

    new_hf_img = new_hf - np.min(new_hf)
    new_hf_img = new_hf_img / np.max(new_hf_img)

    return new_hf_img

def eval_uv(tex, uvs):
    h, w = tex.shape
    _uvs = uvs.copy()
    uf, ui = np.modf(_uvs[:, 0] * w - 0.5)
    vf, vi = np.modf(_uvs[:, 1] * h - 0.5)
    uf[uf < 0] = 1 + uf[uf < 0]
    vf[vf < 0] = 1 + vf[vf < 0]
    ui = np.maximum(ui.astype(int), 0)
    vi = np.maximum(vi.astype(int), 0)
    vals = np.column_stack([tex[vi, ui], tex[vi, np.minimum(ui + 1, w - 1)], tex[np.minimum(vi + 1, h - 1), ui], tex[np.minimum(vi + 1, h - 1), np.minimum(ui + 1, w - 1)]])
    return (1 - vf) * (1 - uf) * vals[:, 0] + (1 - vf) * uf * vals[:, 1] + vf * (1 - uf) * vals[:, 2] + vf * uf * vals[:, 3]

def smoothness(tex_torch):
    data_padded = torch.nn.functional.pad(tex_torch, (1, 1, 1, 1), value=0)
    # minimize squared laplacian
    lap = data_padded[:-2, 1:-1] + data_padded[2:, 1:-1] + data_padded[1:-1, :-2] + data_padded[1:-1, 2:] - 4 * data_padded[1:-1, 1:-1]
    loss = torch.mean(lap * lap)
    return loss

def optimize_tex(tex, edge_border, iterations=100, smoothness_wt=0.1, device='cpu'):
    tex_torch = torch.tensor(tex, device=device)
    orig = torch.nn.functional.pad(tex_torch.clone(), (edge_border, edge_border, edge_border, edge_border), value=0)
    tex_torch.requires_grad = True

    optimizer = torch.optim.Adam([tex_torch], lr=0.01)
    
    for _ in tqdm.trange(iterations):
        optimizer.zero_grad()
        hfield = torch.nn.functional.pad(tex_torch, (edge_border, edge_border, edge_border, edge_border), value=0)
        # apply smoothness loss
        sm = smoothness(hfield) * smoothness_wt
        tx = torch.mean((hfield - orig) ** 2)
        loss = sm + tx
        loss.backward()
        optimizer.step()
    
    optimized_tex = tex_torch.detach().cpu().numpy()
    return optimized_tex

def preprocess_tex(tex_torch, edge_border):
    hfield = np.pad(tex_torch, edge_border, mode="constant", constant_values=0)
    return hfield

def rect_points_to_uv(points, x_min=-0.3, x_max=0.3, z_min=-0.3, z_max=0.3, uvmin=0.0001, uvmax=0.9999):
    return np.column_stack([(points[:, 0] - x_min) / (x_max - x_min) * (uvmax - uvmin) + uvmin, (points[:, 2] - z_min) / (z_max - z_min) * (uvmax - uvmin) + uvmin])

def gen_relief(hmap, out_file, ht=0.05, device='cpu'):
    Ps, Es = mesher.box_mesher(0.02, 0.6, 0.6, 0.15)

    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # subdivide top faces if needed
    top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
    Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, 1)
    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # load image
    ht_low = -ht
    ht_high = ht
    tex = (hmap - np.min(hmap)) / (np.max(hmap) - np.min(hmap)) * (ht_high - ht_low) + ht_low
    tex = optimize_tex(tex, edge_border=2, iterations=100, smoothness_wt=3, device=device)
    tex = preprocess_tex(tex, edge_border=2)
    # save text to tmp
    # import matplotlib.pyplot as plt
    # plt.imsave(f"tmp/{pathlib.Path(out_file).stem}_gray.png", tex, cmap="gray")
    # diffmesh = dm.DiffMesh(Ps, Es)
    # CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]
    # rendered_imgs = [diffmesh.render(tex, *cam, radius=1) for cam in CAM_POS]
    # H.save_images(f"tmp/{pathlib.Path(out_file).stem}_renders.png", rendered_imgs + [tex], auto_grid=True)

    heights = eval_uv(tex, diff_uvs)
    Ps[diff_pts_idx, 1] -= heights
    H.save_mesh(out_file, Ps, Es)

def center_crop(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top+min_dim, left:left+min_dim]

def resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def contrast(
    depth_np,
    method="clip",
    clip_percentiles=(5, 95),
    gamma=None,
    clahe_clip_limit=2.0,
    clahe_tile_grid=(8, 8)
):
    """
    Enhance contrast of a depth map.

    Parameters:
    - depth_np: 2D numpy array (raw depth map)
    - method: "clip", "equalize", or "clahe"
    - clip_percentiles: Tuple of (low, high) percentiles for 'clip' method
    - gamma: Apply optional gamma correction (e.g., 0.8 for boost)
    - clahe_clip_limit: CLAHE clip limit (if using 'clahe')
    - clahe_tile_grid: CLAHE tile grid size (if using 'clahe')

    Returns:
    - depth_contrast: 2D numpy array with values in [0, 1]
    """
    if method == "clip":
        vmin = np.percentile(depth_np, clip_percentiles[0])
        vmax = np.percentile(depth_np, clip_percentiles[1])
        depth = np.clip(depth_np, vmin, vmax)
        depth_contrast = (depth - vmin) / (vmax - vmin)

    elif method == "equalize":
        depth_uint8 = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_uint8 = (depth_uint8 * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(depth_uint8)
        depth_contrast = equalized.astype(np.float32) / 255.0

    elif method == "clahe":
        depth_uint8 = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_uint8 = (depth_uint8 * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        clahe_result = clahe.apply(depth_uint8)
        depth_contrast = clahe_result.astype(np.float32) / 255.0

    else:
        raise ValueError("Unsupported method. Use 'clip', 'equalize', or 'clahe'.")

    # Optional gamma correction
    if gamma is not None:
        depth_contrast = np.power(depth_contrast, gamma)

    return depth_contrast


if __name__ == "__main__":
    import os
    batch_num = 0
    os.makedirs(f"reliefs/batch{batch_num}", exist_ok=True)
    alls = glob.glob(f"images/batch{batch_num}/*.png")
    d = 0
    for img in alls:
        print(f"Processing {img} ({d + 1} out of {len(alls)})")
        d += 1
        hmap = gen_hmap(img)
        n = pathlib.Path(img).stem
        gen_relief(hmap, f"reliefs/batch{batch_num}/{n}.obj", ht=0.05)
