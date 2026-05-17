"""
Double-RIS Wireless-Powered Communication System — Day 2 Simulation (Python)
Paper: "Double-RIS Enabled Physical Layer Security for WPC Systems
        Over Rayleigh Fading Channels" — Kunrui Cao et al., IEEE 2025

Metrics : COP, SOP, EST for all 4 DRIS schemes
Method  : Monte Carlo simulation
Plots   : COP vs P_PS, SOP vs P_PS, EST vs P_PS, COP/SOP vs N
Run     : python DRIS_WPC_simulation.py
Requires: pip install numpy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PARAMETERS  (calibrated: P_PS in dBm, noise = -90 dBm)
# ─────────────────────────────────────────────────────────────────────────────
N          = 32          # Number of RIS reflecting elements
eta        = 0.8         # RF-to-DC energy conversion efficiency
alpha_t    = 0.5         # Time-switching ratio  T_EH / T_total
R_target   = 2.0         # Target data rate (bits/s/Hz) for COP threshold
R_s        = 1.0         # Target secrecy rate (bits/s/Hz) for SOP
M_sim      = int(1e5)    # Monte Carlo iterations
sigma2     = 1e-12       # Noise power in Watts  (~-90 dBm noise floor)
path_exp   = 2.7         # Path loss exponent

# Node distances in metres
d = dict(PS_U=30,PS_J=35,PS_R1=15,R1_U=20,R1_J=22,
         U_AP=25,U_R2=15,R2_AP=20,U_E=50,J_E=45,J_R2=18,R2_E=40)

# Mean channel gains: Omega_ij = d_ij^(-path_exp)
Om = {k: v**(-path_exp) for k, v in d.items()}

gamma_th = 2**R_target - 1       # SNR threshold for COP
t_ratio  = alpha_t / (1-alpha_t) # T_EH / T_IT

rng = np.random.default_rng(42)  # reproducible results

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: complex Rayleigh channel  h ~ CN(0, Omega)
# ─────────────────────────────────────────────────────────────────────────────
def rc(Omega, rows, cols=1):
    s = np.sqrt(Omega / 2)
    return s * (rng.standard_normal((rows,cols)) + 1j*rng.standard_normal((rows,cols)))

# ─────────────────────────────────────────────────────────────────────────────
# POWER SWEEP  (-20 to +20 dBm)
# ─────────────────────────────────────────────────────────────────────────────
P_dBm  = np.arange(-20, 21, 2)
P_lin  = 10.0**(P_dBm/10) * 1e-3   # convert dBm → Watts
n_pts  = len(P_dBm)

COP_MC = np.zeros((n_pts, 4))
SOP_MC = np.zeros((n_pts, 4))
EST_MC = np.zeros((n_pts, 4))

schemes = ['DRIS-1','DRIS-2','DRIS-3','DRIS-4']
colors  = ['#378ADD','#1D9E75','#BA7517','#D85A30']
markers = ['o','s','^','d']

print(f"Running Monte Carlo — {M_sim:.0e} iterations per power level ...")

for p_i, P_PS in enumerate(P_lin):

    # --- Generate channels ---
    h_PU  = rc(Om['PS_U'],  M_sim).ravel()
    h_PJ  = rc(Om['PS_J'],  M_sim).ravel()
    h_UA  = rc(Om['U_AP'],  M_sim).ravel()
    h_UE  = rc(Om['U_E'],   M_sim).ravel()
    h_JE  = rc(Om['J_E'],   M_sim).ravel()
    h_PR1 = rc(Om['PS_R1'], M_sim, N)
    h_R1U = rc(Om['R1_U'],  M_sim, N)
    h_R1J = rc(Om['R1_J'],  M_sim, N)
    h_UR2 = rc(Om['U_R2'],  M_sim, N)
    h_R2A = rc(Om['R2_AP'], M_sim, N)
    h_JR2 = rc(Om['J_R2'],  M_sim, N)
    h_R2E = rc(Om['R2_E'],  M_sim, N)

    # --- Coherent combining gains (optimal RIS phases) ---
    G1U  = np.sum(np.abs(h_PR1)*np.abs(h_R1U), axis=1)  # RIS-1 → U
    G1J  = np.sum(np.abs(h_PR1)*np.abs(h_R1J), axis=1)  # RIS-1 → J
    G2UA = np.sum(np.abs(h_UR2)*np.abs(h_R2A), axis=1)  # RIS-2 → AP
    G2JE = np.sum(np.abs(h_JR2)*np.abs(h_R2E), axis=1)  # RIS-2 → Eve

    # --- Transmit powers per scheme ---
    PU1 = eta*P_PS*(np.abs(h_PU)+G1U)**2 * t_ratio   # DRIS-1/2: RIS-1 helps U
    PJ1 = eta*P_PS*np.abs(h_PJ)**2 * t_ratio          # DRIS-1/2: J no RIS
    PU3 = eta*P_PS*np.abs(h_PU)**2 * t_ratio           # DRIS-3/4: U no RIS
    PJ3 = eta*P_PS*(np.abs(h_PJ)+G1J)**2 * t_ratio    # DRIS-3/4: RIS-1 helps J

    # --- Channel power terms ---
    HAP_r = (np.abs(h_UA)+G2UA)**2   # U→AP with RIS-2
    HAP_n = np.abs(h_UA)**2           # U→AP without RIS-2
    HJE_r = (np.abs(h_JE)+G2JE)**2   # J→Eve with RIS-2 (stronger jam)
    HJE_n = np.abs(h_JE)**2           # J→Eve without RIS-2
    HUE   = np.abs(h_UE)**2           # U→Eve direct

    # DRIS-1: RIS-2→AP(IT), J jams without RIS
    # DRIS-2: RIS-2→Eve(NT), J jams with RIS
    # DRIS-3: RIS-2→AP(IT), J jams without RIS (but J stronger from RIS-1 EH)
    # DRIS-4: RIS-2→Eve(NT), J jams with RIS (and J stronger from RIS-1 EH)
    cfg = [
        (PU1, PJ1, HAP_r, HJE_n),
        (PU1, PJ1, HAP_n, HJE_r),
        (PU3, PJ3, HAP_r, HJE_n),
        (PU3, PJ3, HAP_n, HJE_r),
    ]

    for s, (P_U, P_J, H_AP, H_JE) in enumerate(cfg):
        g_AP = P_U * H_AP / sigma2
        g_E  = P_U * HUE / (P_J*H_JE + sigma2)
        C_AP = np.log2(1 + g_AP)
        C_E  = np.log2(1 + g_E)
        COP_MC[p_i,s] = np.mean(g_AP < gamma_th)
        SOP_MC[p_i,s] = np.mean((C_AP - C_E) < R_s)
        EST_MC[p_i,s] = R_s*(1-SOP_MC[p_i,s])*(1-COP_MC[p_i,s])

    if (p_i+1) % 5 == 0 or p_i == 0:
        print(f"  P_PS = {P_dBm[p_i]:+3.0f} dBm done")

print("Power sweep complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# N SWEEP  at P_PS = -5 dBm
# ─────────────────────────────────────────────────────────────────────────────
N_range    = [4, 8, 16, 32, 64, 128]
P_PS_fixed = 10**(-5/10) * 1e-3
COP_vs_N   = np.zeros((len(N_range), 4))
SOP_vs_N   = np.zeros((len(N_range), 4))

print("Running N sweep ...")
for n_i, Nc in enumerate(N_range):
    h_PU=rc(Om['PS_U'],M_sim).ravel(); h_PJ=rc(Om['PS_J'],M_sim).ravel()
    h_UA=rc(Om['U_AP'],M_sim).ravel(); h_UE=rc(Om['U_E'],M_sim).ravel(); h_JE=rc(Om['J_E'],M_sim).ravel()
    h_PR1=rc(Om['PS_R1'],M_sim,Nc); h_R1U=rc(Om['R1_U'],M_sim,Nc); h_R1J=rc(Om['R1_J'],M_sim,Nc)
    h_UR2=rc(Om['U_R2'],M_sim,Nc);  h_R2A=rc(Om['R2_AP'],M_sim,Nc)
    h_JR2=rc(Om['J_R2'],M_sim,Nc);  h_R2E=rc(Om['R2_E'],M_sim,Nc)

    G1U=np.sum(np.abs(h_PR1)*np.abs(h_R1U),axis=1)
    G1J=np.sum(np.abs(h_PR1)*np.abs(h_R1J),axis=1)
    G2UA=np.sum(np.abs(h_UR2)*np.abs(h_R2A),axis=1)
    G2JE=np.sum(np.abs(h_JR2)*np.abs(h_R2E),axis=1)

    PU1=eta*P_PS_fixed*(np.abs(h_PU)+G1U)**2*t_ratio; PJ1=eta*P_PS_fixed*np.abs(h_PJ)**2*t_ratio
    PU3=eta*P_PS_fixed*np.abs(h_PU)**2*t_ratio;        PJ3=eta*P_PS_fixed*(np.abs(h_PJ)+G1J)**2*t_ratio
    HUE=np.abs(h_UE)**2
    cfg_n = [
        (PU1,PJ1,(np.abs(h_UA)+G2UA)**2, np.abs(h_JE)**2),
        (PU1,PJ1, np.abs(h_UA)**2,        (np.abs(h_JE)+G2JE)**2),
        (PU3,PJ3,(np.abs(h_UA)+G2UA)**2, np.abs(h_JE)**2),
        (PU3,PJ3, np.abs(h_UA)**2,        (np.abs(h_JE)+G2JE)**2),
    ]
    for s,(P_U,P_J,H_AP,H_JE) in enumerate(cfg_n):
        g_AP=P_U*H_AP/sigma2; g_E=P_U*HUE/(P_J*H_JE+sigma2)
        COP_vs_N[n_i,s]=np.mean(g_AP<gamma_th)
        SOP_vs_N[n_i,s]=np.mean((np.log2(1+g_AP)-np.log2(1+g_E))<R_s)
    print(f"  N = {Nc:3d} done")

print("N sweep complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Power sweep (3 subplots)
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1,3,figsize=(15,4.8))
fig1.suptitle(f'Double-RIS WPC PLS — Monte Carlo (N={N}, η={eta}, α={alpha_t}, $R_s$={R_s} b/s/Hz)',fontsize=11)

for s in range(4):
    axes[0].semilogy(P_dBm, COP_MC[:,s], f'-{markers[s]}',
                     color=colors[s],label=schemes[s],lw=1.8,ms=6,markevery=3)
axes[0].set(xlabel='$P_{PS}$ (dBm)',ylabel='COP',title='COP vs Transmit Power')
axes[0].legend(fontsize=9); axes[0].grid(True,which='both',alpha=0.35); axes[0].set_ylim([1e-4,1.1])

for s in range(4):
    axes[1].semilogy(P_dBm,np.maximum(SOP_MC[:,s],1e-5),f'-{markers[s]}',
                     color=colors[s],label=schemes[s],lw=1.8,ms=6,markevery=3)
axes[1].set(xlabel='$P_{PS}$ (dBm)',ylabel='SOP',title='SOP vs Transmit Power')
axes[1].legend(fontsize=9); axes[1].grid(True,which='both',alpha=0.35); axes[1].set_ylim([1e-4,1.1])

for s in range(4):
    axes[2].plot(P_dBm,EST_MC[:,s],f'-{markers[s]}',
                 color=colors[s],label=schemes[s],lw=1.8,ms=6,markevery=3)
axes[2].set(xlabel='$P_{PS}$ (dBm)',ylabel='EST (bits/s/Hz)',title='EST vs Transmit Power')
axes[2].legend(fontsize=9); axes[2].grid(True,alpha=0.35); axes[2].set_ylim([0,R_s*1.05])

plt.tight_layout()
plt.savefig('DRIS_power_sweep.png',dpi=150,bbox_inches='tight')
print("Saved: DRIS_power_sweep.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — N sweep
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1,2,figsize=(10,4.5))
fig2.suptitle(f'Performance vs N  (P_PS=−5 dBm,  $R_s$={R_s} b/s/Hz)',fontsize=11)

for s in range(4):
    axes2[0].semilogy(N_range,np.maximum(COP_vs_N[:,s],1e-5),f'-{markers[s]}',
                      color=colors[s],label=schemes[s],lw=1.8,ms=7)
axes2[0].set(xlabel='N (RIS elements)',ylabel='COP',title='COP vs N')
axes2[0].legend(); axes2[0].grid(True,which='both',alpha=0.35)

for s in range(4):
    axes2[1].semilogy(N_range,np.maximum(SOP_vs_N[:,s],1e-5),f'-{markers[s]}',
                      color=colors[s],label=schemes[s],lw=1.8,ms=7)
axes2[1].set(xlabel='N (RIS elements)',ylabel='SOP',title='SOP vs N')
axes2[1].legend(); axes2[1].grid(True,which='both',alpha=0.35)

plt.tight_layout()
plt.savefig('DRIS_N_sweep.png',dpi=150,bbox_inches='tight')
print("Saved: DRIS_N_sweep.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"RESULTS SUMMARY  (N={N},  R_target={R_target},  R_s={R_s})")
print(f"{'─'*68}")
print(f"{'P(dBm)':>7} | {'COP D1':>7} | {'COP D4':>7} | {'SOP D1':>7} | {'SOP D4':>7} | {'EST D1':>7}")
print(f"{'─'*68}")
for i in range(0, n_pts, 4):
    print(f"{P_dBm[i]:>7.0f} | {COP_MC[i,0]:>7.4f} | {COP_MC[i,3]:>7.4f} | "
          f"{SOP_MC[i,0]:>7.4f} | {SOP_MC[i,3]:>7.4f} | {EST_MC[i,0]:>7.4f}")
print(f"\nExpected ordering:")
print(f"  COP: DRIS-1 lowest (most reliable) → DRIS-4 highest")
print(f"  SOP: DRIS-4 lowest (most secure)  → DRIS-1 highest")
print(f"  EST: DRIS-3 peaks at mid-power (best balanced trade-off)")