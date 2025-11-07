import numpy as np
import matplotlib.pyplot as plt

# 1) DC Motor parametreleri
R = 1.0
L = 0.5
Kt = 0.01
Kb = 0.01
J = 0.01
B = 0.001
Vmax = 24.0


# 2) Üyelik fonksiyonları
def triangular(x, a, b, c):
    x = np.asarray(x)
    mu = np.zeros_like(x, dtype=float)
    left = (a < x) & (x <= b)
    mu[left] = (x[left] - a) / (b - a + 1e-12)
    right = (b < x) & (x < c)
    mu[right] = (c - x[right]) / (c - b + 1e-12)
    mu[x == b] = 1.0
    mu[x == a] = 0.0
    mu[x == c] = 0.0
    return mu

# Hata (e)
e_NB = (-400, -300, -150)
e_NS = (-200, -100, -40)
e_Z  = (-8, 0, 8)
e_PS = (0, 40, 80)
e_PB = (50, 100, 200)

# Hata değişimi (de)
de_N = (-40, -20, 0)
de_Z = (-4, 0, 4)
de_P = (0, 20, 40)

# Çıkış (u) üyelikleri
u_N = (-1.5*Vmax, -Vmax, -Vmax/2)
u_Z = (-6, 0, 6)
u_P = (0, 18, 36)

#  5x3 kural tablosu (e x de)
rule_table = [
    ['N', 'N', 'N'],   # NB
    ['N', 'N', 'Z'],   # NS
    ['N', 'Z', 'P'],   # Z
    ['Z', 'P', 'P'],   # PS
    ['P', 'P', 'P']    # PB
]

output_mfs = {'N': u_N, 'Z': u_Z, 'P': u_P}

# --- Fuzzification ---
def fuzzify_e_de(e, de):
    mu_e = {
        'NB': triangular([e], *e_NB)[0],
        'NS': triangular([e], *e_NS)[0],
        'Z' : triangular([e], *e_Z)[0],
        'PS': triangular([e], *e_PS)[0],
        'PB': triangular([e], *e_PB)[0],
    }
    mu_de = {
        'N': triangular([de], *de_N)[0],
        'Z': triangular([de], *de_Z)[0],
        'P': triangular([de], *de_P)[0],
    }
    return mu_e, mu_de

# Mamdani Defuzzification (centroid)
def mamdani_defuzz(e, de, u_disc=np.linspace(-1.5*Vmax, 1.5*Vmax, 1501)):
    mu_e, mu_de = fuzzify_e_de(e, de)
    aggregated = np.zeros_like(u_disc)
    e_labels = ['NB', 'NS', 'Z', 'PS', 'PB']
    de_labels = ['N', 'Z', 'P']

    for i_e, e_lab in enumerate(e_labels):
        for j_de, de_lab in enumerate(de_labels):
            fire = min(mu_e[e_lab], mu_de[de_lab])
            if fire <= 0:
                continue
            a, b, c = output_mfs[rule_table[i_e][j_de]]
            mu_out = triangular(u_disc, a, b, c)
            aggregated = np.maximum(aggregated, np.minimum(mu_out, fire))

    num = np.sum(u_disc * aggregated)
    den = np.sum(aggregated)
    if den == 0:
        val = 0.0
    else:
        val = num / den
    return float(np.clip(val, -Vmax, Vmax))

# 3) DC Motor dinamiği
def motor_derivatives(x, u, TL=0.0):
    i, w = x
    di = (-R*i - Kb*w + u)/L
    dw = (-B*w + Kt*i - TL)/J
    return np.array([di, dw])

def rk4_step(x, u, dt, TL=0.0):
    k1 = motor_derivatives(x, u, TL)
    k2 = motor_derivatives(x + 0.5*dt*k1, u, TL)
    k3 = motor_derivatives(x + 0.5*dt*k2, u, TL)
    k4 = motor_derivatives(x + dt*k3, u, TL)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# 4) Kapalı çevrim simülasyonu

def simulate(ref_func, T=15.0, dt=0.001, x0=None):
    if x0 is None:
        x = np.array([0.0, 0.0])
    else:
        x = np.array(x0, dtype=float)
    t = np.arange(0, T+dt, dt)
    N = len(t)
    i_hist = np.zeros(N)
    w_hist = np.zeros(N)
    u_hist = np.zeros(N)
    e_hist = np.zeros(N)
    de_hist = np.zeros(N)
    ref_hist = np.zeros(N)
    prev_e = 0.0

    for k in range(N):
        tk = t[k]
        ref = ref_func(tk)
        e = ref - x[1]
        de = e - prev_e

        # Küçük dış yük (disturbance)
        if 6.0 <= tk < 7.5:
            TL = 0.02 * np.exp(-(tk-6.0)*1.5)
        else:
            TL = 0.0

        u = mamdani_defuzz(e, de)
        x = rk4_step(x, u, dt, TL)
        i_hist[k], w_hist[k], u_hist[k], e_hist[k], de_hist[k], ref_hist[k] = x[0], x[1], u, e, de, ref
        prev_e = e
    return t, w_hist, u_hist, e_hist, de_hist, ref_hist

# 5) Örnek simülasyon ve görselleştirme
if __name__ == "__main__":
    ref_val = 100.0  # rad/s

    # Hızlı rampa
    def ref(t):
        return ref_val * (t/0.05) if t < 0.05 else ref_val

    t, w, u, e, de, ref_sig = simulate(ref, T=15.0, dt=0.001)

    # 1) de (hata değişimi) üyelik fonksiyonları
    de_vals = np.linspace(-50, 50, 400)
    mu_de_N = triangular(de_vals, *de_N)
    mu_de_Z = triangular(de_vals, *de_Z)
    mu_de_P = triangular(de_vals, *de_P)

    plt.figure(figsize=(7,3))
    plt.plot(de_vals, mu_de_N, label='N (Negatif)')
    plt.plot(de_vals, mu_de_Z, label='Z (Sıfır)')
    plt.plot(de_vals, mu_de_P, label='P (Pozitif)')
    plt.title('Hata Değişimi (de) Üyelik Fonksiyonları')
    plt.xlabel('Hata Değişimi (de)'); plt.ylabel('Üyelik Derecesi')
    plt.legend(loc='upper right'); plt.grid(True)
    plt.tight_layout()

    # 2) e (hata) üyelik fonksiyonları
    e_vals = np.linspace(-400, 400, 800)
    mu_NB = triangular(e_vals, *e_NB)
    mu_NS = triangular(e_vals, *e_NS)
    mu_Z  = triangular(e_vals, *e_Z)
    mu_PS = triangular(e_vals, *e_PS)
    mu_PB = triangular(e_vals, *e_PB)

    plt.figure(figsize=(7,3))
    plt.plot(e_vals, mu_NB, label='NB')
    plt.plot(e_vals, mu_NS, label='NS')
    plt.plot(e_vals, mu_Z, label='Z')
    plt.plot(e_vals, mu_PS, label='PS')
    plt.plot(e_vals, mu_PB, label='PB')
    plt.title('Hata (e) Üyelik Fonksiyonları')
    plt.xlabel('e'); plt.ylabel('Üyelik Derecesi'); plt.legend(loc='upper right'); plt.grid(True)
    plt.tight_layout()

    # 3) Ana performans grafikleri: Speed / Voltage / Error
    plt.figure(figsize=(7,8))
    plt.subplot(3,1,1)
    plt.plot(t, ref_sig, '--', label='Ref')
    plt.plot(t, w, label='Omega')
    plt.ylabel('Speed (rad/s)')
    plt.legend(loc='upper right'); plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, u, label='Voltage')
    plt.ylabel('Voltage (V)'); plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, e, label='Error')
    plt.ylabel('Error'); plt.xlabel('Time (s)'); plt.grid(True)

    plt.tight_layout()
    plt.show()
