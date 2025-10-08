"""
Ball-and-Beam (pivot A at origin, motor on right)

Uses CLOSED-FORM IK and auto-detects the feasible beam-angle band so the
crank doesn't "stick". Plugged with YOUR dimensions below.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------
# 1) YOUR GEOMETRY (from CAD)
# ------------------------------
l    = 14.63   # A -> B beam length
l_11 = 3.363   # crank length
l_12 = 4.924   # connecting-rod length
h    = 5.927  # motor anchor vertical offset (positive = below pivot)
w    = 15.233   # motor anchor horizontal offset (to the right)

A = np.array([0.0, 0.0])          # beam pivot
M = np.array([w, -h])             # motor anchor (fixed)

ELBOW_PREFERENCE = "down"           # "up" or "down"

# ------------------------------
# 2) Helpers
# ------------------------------
def beam_tip(theta):
    """
    Returns:
      - (2,) if theta is a scalar
      - (N,2) if theta is a vector
    """
    th = np.asarray(theta)
    xy = np.stack((l*np.cos(th), l*np.sin(th)), axis=-1)  # last dim = 2
    return xy


def crank_tip(theta1):
    return M + l_11*np.array([np.cos(theta1), np.sin(theta1)])

def segment_polygon(p0, p1, half_thk=0.10):
    v = p1 - p0
    L = np.hypot(v[0], v[1])
    if L < 1e-9:
        return np.array([p0, p0, p0, p0, p0])
    n = np.array([-v[1]/L, v[0]/L]) * half_thk
    return np.array([p0+n, p1+n, p1-n, p0-n, p0+n])

def solve_theta1_closed_form(theta, prev_theta1):
    """
    Closed-form IK for triangle M-C-B:
      |MB| = d, |MC| = l_11, |CB| = l_12.
    Returns (theta1, reachable_bool).
    """
    B = beam_tip(theta)
    dvec = B - M
    d = np.hypot(dvec[0], dvec[1])

    # Feasibility check
    if d > (l_11 + l_12) or d < abs(l_11 - l_12) or d < 1e-9:
        return prev_theta1, False

    phi = np.arctan2(dvec[1], dvec[0])
    c = (l_11*l_11 + d*d - l_12*l_12) / (2.0*l_11*d)
    c = np.clip(c, -1.0, 1.0)
    alpha = np.arccos(c)

    # Two branches
    cand1 = phi + alpha
    cand2 = phi - alpha

    # Prefer elbow-up or elbow-down
    C1 = crank_tip(cand1)
    C2 = crank_tip(cand2)
    if ELBOW_PREFERENCE.lower() == "up":
        preferred, alt = (cand1, cand2) if C1[1] >= C2[1] else (cand2, cand1)
    else:
        preferred, alt = (cand1, cand2) if C1[1] <= C2[1] else (cand2, cand1)

    # Continuity safeguard
    angdiff = lambda a,b: float(np.angle(np.exp(1j*(a-b))))
    theta1 = preferred if abs(angdiff(preferred, prev_theta1)) <= abs(angdiff(alt, prev_theta1)) + 0.5 else alt
    return float(theta1), True

# ------------------------------
# 3) Compute feasible beam-angle band
# ------------------------------
def d_of_theta(theta):
    B = beam_tip(theta)       # (N,2) or (2,)
    dv = B - M                # broadcasts to (N,2) or (2,)
    return np.hypot(dv[...,0], dv[...,1])

dmin_req, dmax_req = abs(l_11 - l_12), (l_11 + l_12)

grid = np.linspace(-np.pi, np.pi, 4001)  # wide search
mask = (d_of_theta(grid) >= dmin_req) & (d_of_theta(grid) <= dmax_req)

if not np.any(mask):
    raise RuntimeError(
        f"No feasible beam angles for these dimensions. "
        f"Need d(theta) ∈ [{dmin_req:.3f},{dmax_req:.3f}] but it's never satisfied."
    )

# pick the longest contiguous feasible run
runs, start = [], None
for i, ok in enumerate(mask):
    if ok and start is None: start = i
    if (not ok or i == len(mask)-1) and start is not None:
        end = i if not ok else i
        runs.append((start, end))
        start = None
best = max(runs, key=lambda se: se[1]-se[0])
theta_min, theta_max = grid[best[0]], grid[best[1]]
theta_min = np.deg2rad(-15)
theta_max = np.deg2rad(15)

# center & amplitude safely inside the band (avoid edges)
center = 0.5*(theta_min + theta_max)
amp    = 0.45*(theta_max - theta_min)

print(f"Feasible beam-angle band (deg): [{np.degrees(theta_min):.2f}, {np.degrees(theta_max):.2f}]")
print(f"Animating around center={np.degrees(center):.2f}°, amplitude=±{np.degrees(amp):.2f}°")

# ------------------------------
# 4) Plot & animation
# ------------------------------
fig, ax = plt.subplots()
padx = max(6.0, 0.35*l + w)
pady = max(6.0, 0.35*l + h)
ax.set_aspect('equal', 'box')
ax.set_xlim(-padx,  l + w + 0.3*padx)
ax.set_ylim(-(pady + 0.3*l),  0.6*pady)
ax.set_title("Beam pivot at A (origin)")

beam_line, = ax.plot([], [], lw=3, color='blue')
rod_line,  = ax.plot([], [], lw=3, color='green')
crk_line,  = ax.plot([], [], lw=3, color='red')

# Point markers (A,B,C,D)
ptA, = ax.plot([], [], 'ko', ms=6)
ptB, = ax.plot([], [], 'bo', ms=6)
ptC, = ax.plot([], [], 'go', ms=6)
ptD, = ax.plot([], [], 'ro', ms=6)

labA = ax.text(0, 0, "", fontsize=9)
labB = ax.text(0, 0, "", fontsize=9)
labC = ax.text(0, 0, "", fontsize=9)
labD = ax.text(0, 0, "", fontsize=9)

def draw_pose(theta, theta1):
    B = beam_tip(theta)
    C = crank_tip(theta1)

    # draw links
    beam_poly = segment_polygon(A, B, half_thk=0.12)
    rod_poly  = segment_polygon(C, B, half_thk=0.10)
    crk_poly  = segment_polygon(M, C, half_thk=0.10)
    beam_line.set_data(beam_poly[:,0], beam_poly[:,1])
    rod_line.set_data(rod_poly[:,0],  rod_poly[:,1])
    crk_line.set_data(crk_poly[:,0],  crk_poly[:,1])

    # points A,B,C,D
    ptA.set_data([A[0]], [A[1]])
    ptB.set_data([B[0]], [B[1]])
    ptC.set_data([C[0]], [C[1]])
    ptD.set_data([M[0]], [M[1]])

    # labels
    labA.set_position((A[0]+0.2, A[1]+0.2)); labA.set_text("A")
    labB.set_position((B[0]+0.2, B[1]+0.2)); labB.set_text("B")
    labC.set_position((C[0]+0.2, C[1]+0.2)); labC.set_text("C")
    labD.set_position((M[0]+0.2, M[1]+0.2)); labD.set_text("D (motor)")

    # show angles in the title
    ax.set_title(f"Beam pivot at A — θ={np.degrees(theta):.1f}°,  θ₁={np.degrees(theta1):.1f}°")


# Initialize inside the feasible band
theta0 = center
theta1_prev, _ = solve_theta1_closed_form(theta0, prev_theta1=0.0)

def init():
    draw_pose(theta0, theta1_prev)
    return beam_line, rod_line, crk_line

def update(frame):
    # Stay *inside* feasible band so the triangle exists every frame
    theta = center + amp*np.sin(2*np.pi*frame/240.0)

    global theta1_prev
    theta1, ok = solve_theta1_closed_form(theta, prev_theta1=theta1_prev)
    if ok:
        theta1_prev = theta1
    else:
        # if you ever hit an edge numerically, hold last pose
        theta1 = theta1_prev

    draw_pose(theta, theta1)
    return beam_line, rod_line, crk_line

ani = FuncAnimation(fig, update, init_func=init, frames=600, interval=30, blit=False, repeat=True)
plt.show()
