import math

def ref_python_black_scholes(price, strike, t, rate, vol):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    a = math.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1 / math.sqrt(z)

    Se = math.exp(b) * S

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    if (w1 > 0):
        d1 = 0.5 * math.erfc(-w1)
        d1c = 1.0 - d1
    else:
        d1c = 0.5 * math.erfc(w1)
        d1 = 1.0 - d1c
    if (w2 > 0):
        d2 = 0.5 * math.erfc(-w2)
        d2c = 1.0 - d2
    else:
        d2c = 0.5 * math.erfc(w2)
        d2 = 1.0 - d2c
        

    call = P * d1 - Se * d2
    put = Se * d2c - P * d1c
    return (call, put)
