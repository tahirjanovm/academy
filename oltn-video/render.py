import sys, subprocess, numpy as np
from PIL import Image, ImageFilter
src, outdir = sys.argv[1], sys.argv[2]
W, H, FPS, T = 1920, 1080, 30, 10
N = FPS*T
PW, PH = 2112, 1188                      # oversized plate for camera move
img = Image.open(src).convert("RGB")     # 960x1280
y0, y1 = 140, 1180
s = PH/(y1-y0)
fg = img.crop((0, y0, 960, y1)).resize((round(960*s), PH), Image.LANCZOS)
fw = fg.width
ox = 890
fa = np.asarray(fg).astype(np.float32)/255
padL, padR = ox, PW-ox-fw
plate = np.pad(fa, ((0,0),(padL,padR),(0,0)), mode="reflect")
blur = np.asarray(Image.fromarray((plate*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28))).astype(np.float32)/255
xs = np.arange(PW)
feather = 220
m = np.clip(np.minimum((xs-ox)/feather, (ox+fw-xs)/feather), 0, 1)
m = m*m*(3-2*m)
plate = plate*m[None,:,None] + blur*(1-m[None,:,None])
# darken left side for header text, keep warm tone
dark = 0.45 + 0.55*np.clip((xs-250)/900, 0, 1)**1.3
plate *= dark[None,:,None]
lum_plate = plate.mean(2)

# ring geometry (original coords -> plate coords)
def P(x, y): return (ox + x*s, (y-y0)*s)
rcx, rcy = P(478, 562)
rx, ry = 160*s, 320*s
YY, XX = np.mgrid[0:PH, 0:PW].astype(np.float32)
ell = ((XX-rcx)/rx)**2 + ((YY-rcy)/ry)**2
ringmask = np.clip((1.0-ell)/0.15, 0, 1)
metal = ringmask * np.clip((lum_plate-0.35)/0.45, 0, 1)**1.5

# sparkle points: bright local maxima on ring
lu8 = Image.fromarray((lum_plate*255).astype(np.uint8)); mx = np.asarray(lu8.filter(ImageFilter.MaxFilter(25)))
lm = (np.asarray(lu8) == mx) & (ringmask > 0.5) & (lum_plate > 0.75)
pts = np.argwhere(lm)
order = np.argsort(-lum_plate[pts[:,0], pts[:,1]])
pts = pts[order][:14]
rng = np.random.default_rng(7)
phases = rng.random(len(pts)); sizes = rng.uniform(0.6, 1.2, len(pts))

def star(size):
    r = int(60*size); g = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    y, x = g; d = np.hypot(x, y)+1e-3
    core = np.exp(-(d/(3.5*size))**2)
    rays = (np.exp(-(np.abs(y)/(1.2*size))**2) + np.exp(-(np.abs(x)/(1.2*size))**2)) * np.exp(-d/(22*size))
    diag = (np.exp(-(np.abs(x-y)/(1.6*size))**2)+np.exp(-(np.abs(x+y)/(1.6*size))**2))*np.exp(-d/(9*size))*0.35
    glow = np.exp(-(d/(14*size))**2)*0.35
    return np.clip(core+rays*0.9+diag+glow, 0, 2)
stars = [star(z) for z in sizes]

# dust particles (periodic over the loop)
ND = 90
dx = rng.uniform(0, W, ND); dyy = rng.uniform(0, H+80, ND)
dsz = rng.uniform(1.2, 4.5, ND); dbr = rng.uniform(0.15, 0.6, ND)
dloops = rng.integers(1, 3, ND); dwob = rng.uniform(0, 2*np.pi, ND)
warm = np.array([1.0, 0.78, 0.45], np.float32)
vig = None
gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
vig = 1 - 0.35*(((gx-W*0.6)/(W*0.75))**2 + ((gy-H/2)/(H*0.8))**2)
vig = np.clip(vig, 0.5, 1)[..., None]

ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
    "-c:v","libx264","-preset","slow","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",f"{outdir}/master.mp4"], stdin=subprocess.PIPE)

def ease(u): return 0.5-0.5*np.cos(2*np.pi*u)
for i in range(N):
    u = i/N
    fr = plate.copy()
    # light sweep across the metal (first 45% of loop)
    sp = u/0.45
    if sp < 1:
        pos = -400 + sp*(rx*2+800) + (rcx-rx)
        band = np.exp(-(((XX - pos) + (YY-rcy)*0.55)/70)**2)
        fr += (metal*band)[...,None]*warm*0.9
    # warm breathing light
    fr *= 1 + 0.04*np.sin(2*np.pi*u)
    # sparkles
    for (py, px), ph, st in zip(pts, phases, stars):
        a = max(0.0, np.sin(2*np.pi*(2*u+ph)))**6
        if a < 0.02: continue
        r = st.shape[0]//2
        ya, yb, xa, xb = py-r, py+r+1, px-r, px+r+1
        if ya<0 or xa<0 or yb>PH or xb>PW: continue
        fr[ya:yb, xa:xb] += (st*a)[...,None]*np.array([1.0,0.93,0.78],np.float32)
    # camera: slow push-in + drift, seamless
    z = 1/(1.10) * (1 + 0.05*ease(u))
    cw, ch = W/z/ (PW/W) * (PW/W), None
    cw = PW/(1.10)/ (1+0.05*ease(u)); ch = cw*H/W
    cx = PW/2 + 25*np.sin(2*np.pi*u) + 30; cy = PH/2 - 10*np.cos(2*np.pi*u)
    box = (cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2)
    im = Image.fromarray((np.clip(fr,0,1)*255).astype(np.uint8)).resize((W,H), Image.BICUBIC, box=box)
    f = np.asarray(im).astype(np.float32)/255
    # dust
    for k in range(ND):
        yy = (dyy[k] - u*(H+80)*dloops[k]) % (H+80) - 40
        xx = dx[k] + 18*np.sin(2*np.pi*u*dloops[k] + dwob[k])
        r = int(dsz[k]*3)+2
        ya, yb, xa, xb = int(yy)-r, int(yy)+r+1, int(xx)-r, int(xx)+r+1
        if ya<0 or xa<0 or yb>H or xb>W: continue
        g = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d2 = (g[0]-yy)**2 + (g[1]-xx)**2
        tw = 0.6+0.4*np.sin(2*np.pi*(u*3*dloops[k])+dwob[k])
        f[ya:yb, xa:xb] += (np.exp(-d2/(dsz[k]**2))*dbr[k]*tw)[...,None]*warm
    f *= vig
    # fade from/to black-ish not needed: loop is seamless
    f += rng.normal(0, 0.012, (H, W, 1)).astype(np.float32)
    ff.stdin.write((np.clip(f,0,1)*255).astype(np.uint8).tobytes())
    if i == 40: im.save(f"{outdir}/dbg.png")
    if i == 75: Image.fromarray((np.clip(f,0,1)*255).astype(np.uint8)).save(f"{outdir}/poster.jpg", quality=92)
ff.stdin.close(); ff.wait()
print("sparkles", len(pts))
