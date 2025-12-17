import numpy as np
import warnings 


# S is big Theta in Rai and Tripathi's paper, theta is small theta

def theta_(h, theta_r, theta_s, n, m, alpha) :
    """Calculates volumetric soil moisture

    Args:
        h (float) : water pressure head
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter
    
    Returns:
        theta (float) : volumetric soil moisture
    """
    if h >= 0:
        return theta_s

    return theta_r + (theta_s - theta_r) * ((1 + np.abs(alpha * h)**n)**(-m))

def S_(theta, theta_r, theta_s) :
    """Calculates relative saturation

    Args:
        theta (float) : volumetric soil moisture
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture

    Returns:
        S (float) : relative saturation
    """
    return (theta - theta_r) / (theta_s - theta_r)

def h_(theta, theta_r, theta_s, n, m, alpha) :
    """Calculates water pressure head

    Args:
        theta (float) : volumetric soil moisture
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter
    
    Returns:
        h (float) : water pressure head
    """
    s = S_(theta, theta_r, theta_s)
    if s >= 1:
        warnings.warn("S=1 and h cannot be determined")
        return 0
    
    return -((s**(-1 / m) - 1)**(1 / n)) / alpha

def k_(h, S, k_s, m) :
    """Calculates hydraulic conductivity

    Args:
        h (float) : water pressure head
        S (float) : relative saturation
        k_s (float) : saturated hydraulic conductivity
        m (float) : van Genuchten model parameter
    
    Returns:
        K (float) : hydraulic conductivity
    """
    if h >= 0 :
        return k_s

    return k_s * np.sqrt(S) * ((1 - (1 - S**(1/m))**m)**2)

def C_(h, theta_r, theta_s, n, m, alpha) :
    """Calculates water capacity (dtheta/dh)

    Args:
        h (float) : water pressure head
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter

    Returns:
        C (float) : water capacity
    """
    if h >= 0 :
        return 10**(-20)
    
    return 10**(-20) + ((theta_s - theta_r) * m * n * alpha * (np.abs(alpha * h)**(n - 1))) / ((1 + np.abs(alpha * h)**n)**(m + 1))



def dkdS(S, k_s, m) :
    S_factor = 1 - (1 - S**(1/m))**m
    return 2 * k_s * np.sqrt(S) * S_factor * ((1 - S**(1/m))**(m - 1) * (S**(1/m - 1))) + (k_s / 2) * (1 / np.sqrt(S)) * (S_factor**2)

def dSdh(S, h, alpha, m) :
    B0 = - alpha * m * np.sign(h) / (1 - m)
    return B0 * (S**((1 + m) / m)) * ((S**(-1/m) - 1)**m)

def dhdtheta(S, h, alpha, m, theta_r, theta_s) :
    return 1/dSdh(S, h, alpha, m)/(theta_s - theta_r)

def d2hdzdtheta(dthetadz, h, S, alpha, m, theta_r, theta_s) :
    B0 = - alpha * m * np.sign(h) / (1 - m)
    return (1/B0/((theta_s - theta_r)**2) ) * dthetadz * (-((1 + m) / m) * S**(-(1 + 2 * m) / m) * ((S**(-1/m) - 1)**(-m)) + S**(-(2 + 2 * m) / m) * ((S**(-1/m) - 1)**(-m - 1)))

def term3(dthetadz, S, k_s, m, theta_r, theta_s) :
    return dkdS(S, k_s, m) * dthetadz / (theta_s - theta_r)

def term1(dthetadz, S, k_s, h, alpha, m, theta_r, theta_s) :
    return term3(dthetadz, S, k_s, m, theta_r, theta_s) * dhdtheta(S, h, alpha, m, theta_r, theta_s) * dthetadz

def term2(dthetadz, d2thetadz2, S, h, alpha, m, theta_r, theta_s) :
    return d2hdzdtheta(dthetadz,h, S, alpha, m, theta_r, theta_s) * dthetadz + dhdtheta(S, h, alpha, m, theta_r, theta_s) * d2thetadz2


