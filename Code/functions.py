import numpy as np
import at
from scipy import signal
import sympy as sp

def response_matrices(lat, dkick, offset, read = 0):
    # Obtaining Ids
    cor_ids = at.get_refpts(lat, at.elements.Corrector)
    bpm_ids = at.get_refpts(lat, at.elements.Monitor)
    quad_ids = at.get_refpts(lat, at.elements.Quadrupole)

    if read:
        # Read from file
        A = np.loadtxt('rings/A.out')
        B = np.loadtxt('rings/B.out')

    else: 
        # Matrix calculation
        a = []
        for quad in quad_ids:
            at.shift_elem(lat[quad], deltax=offset, deltaz=0.0, relative=False)
            orbit0, orbit = at.find_orbit(lat, refpts = bpm_ids)
            x = orbit[:,0]
            y = orbit[:,2]
            at.shift_elem(lat[quad], deltax=0.0, deltaz=0.0, relative=False)
            a.append(x)
        A = np.squeeze(a) / dkick

        b = []
        for cor in cor_ids:
            lat[cor].KickAngle = [dkick, 0.00]
            orbit0, orbit = at.find_orbit(lat, refpts = bpm_ids)
            x = orbit[:,0]
            y = orbit[:,2]
            lat[cor].KickAngle = [0, 0.00]
            b.append(x)
        B = np.squeeze(b) / dkick

        # Results to a file
        np.savetxt('rings/A.out', A, fmt='%1.20f')
        np.savetxt('rings/B.out', B, fmt='%1.20f')

    return A, B

def svd_solve(A, nsv):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = 1./s
    sinv[nsv:] = 0
    Ainv = np.dot(v.transpose(), np.dot(np.diag(sinv), u.transpose()))
    return Ainv

def change_magnets_alignment(lat, ma):
    quad_ids = at.get_refpts(lat, at.elements.Quadrupole)

    i=0
    for quad in quad_ids:   
        at.shift_elem(lat[quad], deltax=ma, deltaz=0.0, relative=False)  
        i = i + 1   

def change_correctors_kick_angle(lat, cs):
    cor_ids = at.get_refpts(lat, at.elements.Corrector)

    i=0
    for cor in cor_ids:   
        lat[cor].KickAngle  = [cs[i], 0]
        i = i + 1

def sinesweep(t, fmin, fmax, amplitude=1, which='linear'):
    if which == 'linear':
        ft = 2*fmin*t+(fmax-fmin)/t[-1]*0.5*(t**2)
    elif which == 'exp':
        ft = fmin*t[-1]/np.log(fmax/fmin)*(np.exp(-t)-1)
    else:
        raise ValueError("Last argummet (which) should be 'exp' or 'linear', "
                         "'{}' given".format(which))

    return np.sin(2*np.pi*ft)

def apply_f(numx, denx, u, x, Ts):
    num = np.array(numx)
    den = np.array(denx) 
    
    if den.size == 1 and num.size == 1:
        return u*num[0]/den[0], x
    if type(u) is not np.ndarray:
            u = np.array([[u]]).T
    else:
        if u.ndim == 1:
            u = u.reshape((u.size, 1))
        elif u.shape[1] != 1:
            u = u.T

    A, B, C, D = signal.tf2ss(numx, denx)
    # A_t, B_t, C_t, D_t = signal.tf2ss(numx, denx)
    # (A, B, C, D, _) = signal.cont2discrete((A_t, B_t, C_t, D_t), Ts, method='bilinear')
    
    A = np.kron(np.eye(u.size), A)
    B = np.kron(np.eye(u.size), B)
    C = np.kron(np.eye(u.size), C)
    D = np.kron(np.eye(u.size), D)

    x_vec = x.reshape((x.size, 1))
    x1_vec = A.dot(x_vec) + B.dot(u)
    y = C.dot(x_vec) + D.dot(u)

    # put back in same order
    if type(u) is not np.ndarray:
        y = y[0, 0]
    else:
        if u.ndim == 1:
            y = y.reshape(y.size)
        elif u.shape[1] != 1:
            y = y.T
    if np.any(abs(y.imag) > 0):
        print('y has complex part {}'.format(y))
        print((A, B, C, D))
        
    return y.reshape(y.size).real, x1_vec.reshape(x.shape)

def poly_from_sympy(xpr, symbol='s'):
    """ Convert Sympy transfer function polynomial to Scipy LTI """
    s = sp.Symbol(symbol)
    num, den = sp.simplify(xpr).as_numer_denom()  # expressions
    p_num_den = sp.poly(num, s), sp.poly(den, s)  # polynomials
    c_num_den = [p.all_coeffs() for p in p_num_den]  # coefficients

    # convert to floats
    l_num, l_den = [sp.lambdify((), c)() for c in c_num_den]
    return l_num, l_den

def PID_transfer_function(Kp = 0, Ki = 0, Kd = 0):
    symbol = 's'
    s = sp.Symbol(symbol)
    nump = [Kp]
    denp = [1]
    G = sp.Poly(nump, s) / sp.Poly(denp, s)
    if Ki != 0:
        numi = [Ki]
        deni = [1, 0]
        G = G + (sp.Poly(numi, s) / sp.Poly(deni, s))
    if Kd != 0:
        numd = [Kd, 0]
        dend = [Kd/8, 1]
        G = G + (sp.Poly(numd, s) / sp.Poly(dend, s))

    num_pid, den_pid = poly_from_sympy(G, symbol='s')
    return num_pid, den_pid

def real_perturbation(t):
    N = t.size
    Fs = 1/(t[1]-t[0])
    freqs = np.fft.fftfreq(N, 1/Fs)
    freqs_half = freqs[:N//2+1]
    cm_fft = 5*np.random.random(N//2+1)*np.exp(1j*2*np.pi*np.random.random(N//2+1))

    idxmin = np.argmin(abs(freqs_half - 9))
    idx20 = np.argmin(abs(freqs_half - 20))
    for k in range(idxmin, idx20):
        cm_fft[k] = 0.1*cm_fft[k]*(5 - (freqs_half[k] - 11)*(freqs_half[k] - 20))

    nprand = np.random.random
    cmph10 = 2*np.pi*nprand()
    cm_fft[np.argmin(abs(freqs_half - 0))] = 0
    cm_fft[np.argmin(abs(freqs_half - 10))] = 20*np.exp(1j*cmph10)
    cm_fft[np.argmin(abs(freqs_half - 50))] = 30*np.exp(1j*2*np.pi*nprand())
    cm_fft[-1] = 0

    cm_fft = np.concatenate((cm_fft[:-1], np.flipud(cm_fft.conjugate())[:-1]))
    cm_fft *= N/2/np.max(np.abs(cm_fft))
    cm = np.fft.ifft(cm_fft).real

    return cm
