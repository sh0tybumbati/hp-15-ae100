import trimesh, numpy as np, sys
from PIL import Image, ImageDraw
def render(m, elev, azim, W=900, H=600, base=(80,130,210), zoom_center=None, zoom=1.0, fcolors=None):
    """fcolors: optional (n_faces, 3) array of per-face base colours (for multi-part scenes)."""
    v=m.vertices.copy(); c=m.bounds.mean(0) if zoom_center is None else np.array(zoom_center)
    v-=c
    a=np.radians(azim); e=np.radians(elev)
    Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(e-np.pi/2),-np.sin(e-np.pi/2)],[0,np.sin(e-np.pi/2),np.cos(e-np.pi/2)]])
    R=Rx@Rz
    p=v@R.T            # x right, y up(ish), z toward/away
    ext=np.abs(p[:,:2]).max()
    s=min(W,H)*0.46/ext*zoom
    sx=p[:,0]*s+W/2; sy=H/2-p[:,1]*s; sz=p[:,2]
    n=m.face_normals@R.T
    L=np.array([0.3,0.5,0.8]);L/=np.linalg.norm(L)
    sh=np.clip(n@L,0,1)*0.65+0.35
    zb=np.full((H,W),-1e9); img=np.full((H,W,3),255,np.uint8)
    for fi,(f,shade,nz) in enumerate(zip(m.faces,sh,n[:,2])):
        if nz<=0: continue
        xs=sx[f];ys=sy[f];zs=sz[f]
        x0,x1=int(max(xs.min(),0)),int(min(xs.max()+1,W-1));y0,y1=int(max(ys.min(),0)),int(min(ys.max()+1,H-1))
        if x1<x0 or y1<y0: continue
        d=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
        if abs(d)<1e-9: continue
        X,Y=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
        l0=((ys[1]-ys[2])*(X-xs[2])+(xs[2]-xs[1])*(Y-ys[2]))/d
        l1=((ys[2]-ys[0])*(X-xs[2])+(xs[0]-xs[2])*(Y-ys[2]))/d
        l2=1-l0-l1
        mk=(l0>=-1e-6)&(l1>=-1e-6)&(l2>=-1e-6)
        if not mk.any(): continue
        Z=l0*zs[0]+l1*zs[1]+l2*zs[2]
        sub=zb[y0:y1+1,x0:x1+1]; ok=mk&(Z>sub)
        sub[ok]=Z[ok]; img[y0:y1+1,x0:x1+1][ok]=(np.array(base if fcolors is None else fcolors[fi])*shade).astype(np.uint8)
    return Image.fromarray(img)
if __name__=="__main__":
    out="model_preview.png"
    b=trimesh.load("out/case_base.stl"); l=trimesh.load("out/case_lid.stl")
    z_top=b.bounds[1][2]-2.0                       # rim height (tongue sticks 2 mm above)
    l2=l.copy(); l2.apply_translation([0,0,z_top])
    asm=trimesh.util.concatenate([b,l2])
    tiles=[(render(asm,28,-35),"Assembled: front-right"),
           (render(asm,6,0,zoom=1.0),"Assembled: side elevation (stepped profile)"),
           (render(b,35,-35),"Base (ports, fluted walls, stepped foot)"),
           (render(l,35,-35,base=(215,150,70)),"Lid (prints underside-down, no supports)")]
    W,H=tiles[0][0].size; sheet=Image.new("RGB",(W*2,H*2+0),"white"); d=ImageDraw.Draw(sheet)
    for i,(im,t) in enumerate(tiles):
        sheet.paste(im,((i%2)*W,(i//2)*H)); d.text(((i%2)*W+10,(i//2)*H+8),t,fill=(0,0,0))
    sheet.save(out); print(len(b.faces),len(l.faces))
