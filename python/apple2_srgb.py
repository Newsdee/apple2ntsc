"""
Apple II NTSC color conversion routines
  - original GNU bc code and maths by Linard Ticmanis 
  - python port by Newsdee
"""
import math

# You can set the precision here. The math is using exact values only,
# and thus should work at any precision
GLOBALSCALE = 34
SCALE = GLOBALSCALE
S1 = int(GLOBALSCALE / 2.0)+5
S2 = int(GLOBALSCALE / 2.0)

# py porting note: this is orignally an 'auto' bc variable; we will treat it as module-level var
OS_NEWTON = SCALE

def sqr(x):
    """square"""
    return x * x

def cbrt(x):
    """cubic root"""
    return x**(1./3.)    

def odd(n):
    '''check if a number is odd, i.e. lowest bit is set n binary format'''
    return (n % 2) == 0

def sar(n):
    """shift arithmetic right"""
    return int(n/2)

def strhex(n):
    """print two-digit hex value between 00 and ff"""
    return '{0:02x}'.format(int(n)).upper()

def ansi_rgb(r, g, b):
    """returns escape ANSI code for a given RGB value"""
    return '\x1b[48;2;' + str(int(r)) + ';' + str(int(g)) + ";" + str(int(b)) + 'm'

def clamp(x):
    """clamp value to something between 0.0 and 1.0"""
    if x < 0: 
        return 0
    if x > 1: 
        return 1
    return x

# Degrees-based trigonometry
C_180_OVER_PI = 180. / math.pi
C_PI_OVER_180 = math.pi / 180.
def darctan2(x,y):
    """arctan2 in degrees"""
    return C_180_OVER_PI * math.atan2(x, y)
def dcos(x):
    """cos in degrees"""
    return math.cos(C_PI_OVER_180*x)
def dsin(x):
    """sin in degrees"""
    return math.sin(C_PI_OVER_180*x)

# Undo gamma correction, according to SMPTE-170M gamma formula.
# But derive the constants directly to get full precision.
# beta is defined as the root of f(x) near 0.018, this provides
# for a smooth joining of the linear and gamma parts of the
# function, i.e. both parts will have the same value and the same
# first derivative value at that point. 
def f(x):
    '''f of x'''
    return 11/2. * x - 10. * math.pow(x, 11/20) + 1
def f_prime(x):
    '''f prime of x'''
    return 11/2. - (11. / (2. * math.pow(x, 9./20.)))
def _newton(x):
    """newton method to find the root, searching near x"""
    if f(x) / 9.:
        # scale = os_newton + 100.  - py port note: not sure of the usage of this var
        d = x - f(x) / f_prime(x)
        # scale = os_newton
        return _newton(d)
    return x
def newton(x):
    '''newton root of x'''
    # os_newton = scale
    return _newton(x) / 1.0
beta = newton(0.018)

# use variables to hold precalculated constants, to speed up function
alpha = 10. * math.pow(beta, 11./20.)
alpha_minus_1 = alpha - 1.0
delta = 9. / 2. * beta
TWO_OVER_NINE = 2. / 9.
TWENTY_OVER_NINE = 20. / 9.
def linearize_ntsc(x):
    '''linearization of ntsc value'''
    if x < delta :
        return x * TWO_OVER_NINE
    return pow( (x + alpha_minus_1) / alpha, TWENTY_OVER_NINE)

# Do gamma correction, according to sRGB gamma formula, which has a precise definition.
DELTA_SRGB = 0.04045 / 12.92
POW_SRGB = 5. / 12.
def gammaize_srgb(x):
    '''gamma handling'''
    if x < DELTA_SRGB:
        return 12.92 * x
    return 1.055 * pow(x, POW_SRGB) - 0.055

def xyz_to_x (x, y, z):
    '''correction for x'''
    fsum = x + y + z
    if fsum == 0:
        return 0.3127
    return x / fsum

def xyz_to_y (x, y, z):
    '''correction for y'''
    fsum = x + y + z
    if fsum == 0:
        return 0.3290
    return y / fsum

# precalculate some constants needed later
pow_25_7 = pow(25, 7)

# These factors guarantee that the "outermost" RGB values in the 0.0 to 1.0
# range will overshoot or undershoot the Y range of an NTSC signal by
# exactly 1/3 (0.3333...), i.e. so-called 100% color bars in yellow, cyan,
#  red, and blue will always be within -1/3...4/3 at all points of their
#  sine waves 
ufac = math.sqrt(865318419. / 209556997.)
vfac = math.sqrt(288439473. / 221990474.)

# Matrix for going from CIE1931 XYZ to linear sRGB. 
# Thanks to Wolfram Alpha for inverting the standard sRGB Matrix for us.
a = [0.] * 9
a[0] =  67119136.
a[1] = -31838320.
a[2] = -10327488.
a[3] = -20068284.
a[4] =  38850255.
a[5] =    859902.
a[6] =   1153856.
a[7] =  -4225640.
a[8] =  21892272.
for i in range(0, 9):
    a[i] = a[i] * 125. / 2588973042.


# Matrix for going from SMPTE-170M to CIE1931 XYZ. 
# Thanks to Bruce Lindbloom and Wolfram Alpha 
zz = [0.] * 9
zz[0] = 51177./130049.
zz[1] = 4987652./13655145.
zz[2] = 5234753./27310290.
zz[3] = 82858./390147.
zz[4] = 1367582./1950735.
zz[5] = 168863./1950735.
zz[6] = 2437./130049.
zz[7] = 1528474./13655145.
zz[8] = 5234753./5462058.

# Here come the colors used by the Apple IIGS for Apple //e  emulation modes. 
# The Apple IIe Card for LC Mac models uses the same values 
# (but inconsistently converted from 4 bits per channel to 16 bits per channel), by the way.
# See "IIGS Technical Note #63" from Apple, and the 'clut' resource "Apple IIe Colors" 
# in the IIe card's "IIe Startup" 68K Mac executable version 2.2.1d.
rc = [0.]*16
gc = [0.]*16
bc = [0.]*16
rc[0] = 0x0; gc[0] = 0x0; bc[0] = 0x0
rc[1] = 0xD; gc[1] = 0x0; bc[1] = 0x3
rc[2] = 0x0; gc[2] = 0x0; bc[2] = 0x9
rc[3] = 0xD; gc[3] = 0x2; bc[3] = 0xD
rc[4] = 0x0; gc[4] = 0x7; bc[4] = 0x2
rc[5] = 0x5; gc[5] = 0x5; bc[5] = 0x5
rc[6] = 0x2; gc[6] = 0x2; bc[6] = 0xF
rc[7] = 0x6; gc[7] = 0xA; bc[7] = 0xF
rc[8] = 0x8; gc[8] = 0x5; bc[8] = 0x0
rc[9] = 0xF; gc[9] = 0x6; bc[9] = 0x0
rc[10] = 0xA; gc[10] = 0xA; bc[10] = 0xA
rc[11] = 0xF; gc[11] = 0x9; bc[11] = 0x8
rc[12] = 0x1; gc[12] = 0xD; bc[12] = 0x0
rc[13] = 0xF; gc[13] = 0xF; bc[13] = 0x0
rc[14] = 0x4; gc[14] = 0xF; bc[14] = 0x9
rc[15] = 0xF; gc[15] = 0xF; bc[15] = 0xF

# Reference white point for both SMPTE-170M and sRGB is standard
# illuminant D65, defined in both norms as: x=0.3127 y=0.3290

# Convert white point from xyY (with Y=1) to XYZ
XW = 0.3127 / 0.3290
YW = 1.0
ZW = (1.0 - 0.3127 - 0.3290) / 0.3290
# functions for converting from XYZ to L*a*b*
def fxyz(val):
    '''convert xyz'''
    if val > 216./24389.:
        return cbrt(val)
    return (24389. / 27. * val + 16.) / 116.
def _fx(xval):
    return fxyz(xval / XW)
def _fy(yval):
    return fxyz(yval / YW)
def _fz(zval):
    return fxyz(zval / ZW)

def limitprod(x, yval):
    """go from "analog" 0.0 to 1.0 value to "digital" 8 or 16 bit value"""
    x = int(clamp(x) * yval)
    if x == yval:
        x -= 1
    return x

# convert Apple IIGS colors into CIE L*a*b* color space
riigs = [0.] * 16
giigs = [0.] * 16
biigs = [0.] * 16
xiigs = [0.] * 16
yiigs = [0.] * 16
y_cap_iigs = [0.] * 16
l_star = [0.] * 16
a_star = [0.] * 16
b_star = [0.] * 16
c_star = [0.] * 16
for c in range(0, 16):
    r = rc[c] / 15.
    g = gc[c] / 15.
    b = bc[c] / 15.
    r = linearize_ntsc(r)
    g = linearize_ntsc(g)
    b = linearize_ntsc(b)
    xl =  zz[0] * r + zz[1] * g + zz[2] * b
    yl =  zz[3] * r + zz[4] * g + zz[5] * b
    zl =  zz[6] * r + zz[7] * g + zz[8] * b
    xiigs[c] = xyz_to_x(xl, yl, zl)
    yiigs[c] = xyz_to_y(xl, yl, zl)
    y_cap_iigs[c] = yl
    fx = _fx(xl)
    fy = _fy(yl)
    fz = _fz(zl)
    l_star[c] = 116 * fy - 16
    a_star[c] = 500 * (fx - fy)
    b_star[c] = 200 * (fy - fz)
    c_star[c] = math.sqrt(sqr(a_star[c]) + sqr(b_star[c]))
    # Calculate sRGB version of IIGS colors for direct comparison
    r = a[0] * xl + a[1] * yl + a[2] * zl
    g = a[3] * xl + a[4] * yl + a[5] * zl
    b = a[6] * xl + a[7] * yl + a[8] * zl
    # do gamma correction
    r = gammaize_srgb(r)
    g = gammaize_srgb(g)
    b = gammaize_srgb(b)
    # restrict to 0-1 range and store final values as 8 bits for
    # use in ANSI terminal escape sequence below
    riigs[c] = limitprod(r, 256)
    giigs[c] = limitprod(g, 256)
    biigs[c] = limitprod(b, 256)


# Square root of 2 Over Pi, which is the first (base freqeuency) periodic term of the fourier
# series expansion of Apple II pulse signal shape, i.e. 1/4 or 3/4 duty cycle rectangular wave.
# For the 4 colorful Hi-Res colors, which are 1/2 duty cycle (square) rectangular waves,
# it's actually 2/pi but that is taken care of by setting BOTH u and v to +/- sr2op, which
# makes the TOTAL vector length 2/pi according to the Theorem of Pythagoras, just as it should be.
sr2op = math.sqrt(2.0) / math.pi

# Calculate Y, U, V for the Apple II colors from signal shape
# (assuming it to be perfectly rectangular)
y = [0.] * 16
u = [0.] * 16
v = [0.] * 16
for c in range(16):
    # start with Y = U = V = 0
    cy = 0.0
    cu = 0.0
    cv = 0.0
    # add up Y, I, and V for the "basic" colors contained 
    # in a given color's 4-bit pattern (i.e. the dark colors)
    h = c
    if odd(h):
        cy += 0.25
        cv += sr2op
    h = sar(h)
    if odd(h):
        cy += 0.25
        cu += sr2op
    h = sar(h)
    if odd(h):
        cy += 0.25
        cv -= sr2op
    h = sar(h)
    if odd(h):
        cy += 0.25
        cu -= sr2op
    # handle NTSC pedestal and the fact that the monitor expects it,
    # but the Apple hardware does not provide it.
    y[c] = (cy - 0.075) / 0.925
    u[c] = cu / 0.925
    v[c] = cv / 0.925

# Starting values for the four NTSC monitor knobs. You can put
# the better values in the comments here to make it converge faster.
BRIGHTNESS =  0.045851297039046863657017 #0 #  0.045851297039046863657017
PICTURE    =  0.892080981320251448772421 # 1 #  0.892080981320251448772421
COLOR      =  0.784866029442122319724180 # 1 #  0.784866029442122319724180
HUE        =  -0.645936288431288302486169 # 0 # -0.645936288431288302486169

# Helper values, pre-initialized sto prevent nonsense result at start
MINERR     = 15*15*16

def print_result(brightness, picture, color, hue, err, MAXTRY,
                 xc, yc, ycap_c, rd, gr, bl, 
                 riigs, giigs, biigs, 
                 xiigs, yiigs, y_cap_iigs, delta_e):
    """prints the result of calculation"""
    print('')
    print ("brightness= %.8f" % brightness)
    print ("picture = %.8f" % picture)
    print ("color = %.8f" % color)
    print ("hue = %.8f" % hue)
    print ("RMS ∆E = %.8f" % err)
    print ("max tries = %d" % MAXTRY)
    print('')
    print ("BASIC       CIE 1931 xyY Apple //e     sRGB    //e   IIgs    sRGB     CIE 1931 xyY Apple IIgs    CIEDE2000")
    print ("--------    -------------------------  ------- ----- ------  -------- -----------------------    ---------")
    for c in range(16):
        line = ''
        line += 'COLOR=%d' % c
        if c < 10:
            line += ' '
        line += '    x=%.3f' % xc[c]
        line += '  y=%.3f' % yc[c]
        line += '  Y=%.3f' % ycap_c[c]
        line += '  #\x1b[31m' + strhex(rd[c])
        line += '\x1b[32m' + strhex(gr[c])
        line += '\x1b[34m' + strhex(bl[c])
        line += '\x1b[39m '
        # Directly display colors for calculated and for
        # IIGS color values using some ANSI escape magic.
        line += ansi_rgb(rd[c], gr[c], bl[c])
        line += '      '
        line += ansi_rgb(riigs[c], giigs[c], biigs[c])
        line += '      '
        line += '\x1b[49m '
        line += ' #\x1b[31m' + strhex(riigs[c])
        line += '\x1b[32m' + strhex(giigs[c])
        line += '\x1b[34m' + strhex(biigs[c])
        line += '\x1b[39m '
        line += ' x=%.3f' % xiigs[c]
        line += ' y=%.3f' % yiigs[c]
        line += ' Y=%.3f' % y_cap_iigs[c]
        # display per-color Delta-E value
        line += '    ∆E=%.3f' % delta_e[c]
        print (line)

def main_search(BRIGHTNESS, PICTURE, COLOR, HUE):
    """
    Main search loop. Find the best possible match for Apple IIgs colors
    that you can get from turning the four knobs of a standard
    NTSC monitor fed with an Apple //e type signal.
    """
    MAXTRY = 7
    MINERR = 15.*15.*16.*100.
    brightness = BRIGHTNESS
    picture = PICTURE
    color = COLOR
    hue = HUE
    attempt = MAXTRY - 8
    delta_e = [0.] * 16
    xc = [0.] * 16
    yc = [0.] * 16
    ycap_c = [0.] * 16
    rd = [0.] * 16
    gr = [0.] * 16
    bl = [0.] * 16
    #while attempt <= (GLOBALSCALE + 1)*8:
    for loops in range(10):
        old_brightness = brightness
        old_picture = picture
        old_color = color
        old_hue = hue
        # attempt == -1 means calculate baseline
        if attempt < 0:
            # no need to calculate baseline more than once
            attempt = 0
            inc = 0
        else:
            # The algorithm proceeds as follows:
            #    1. Alternate between increasing and decreasing a digit
            #    2. Cycle between the four knobs
            #    3. Go to successively less significant digits
            # Any change that doesn't imrpove the result is reversed and the next try is started;
            # if it does improve, use new values as the next baseline,
            # and go back eight tries (one digit).
            inc = pow(10, -int(attempt/8.))
            if odd(attempt):
                inc = -inc
            # py port note:
            # this should be a switch/case but not supported untl Python 3.10
            tcase = int(attempt % 8) / 2
            if tcase == 0:
                brightness += inc
            if tcase == 1:
                picture += inc
            if tcase == 2:
                color += inc
            if tcase == 3:
                hue += inc
        err = 0
        # loop over the lo-res color numbers
        for c in range(16):
            # rotate U, V by hue
            u0 = u[c]
            v0 = v[c]
            cu = u0 * dcos(hue) - v0 * dsin(hue)
            cv = v0 * dcos(hue) + u0 * dsin(hue)
            # calc R, G, B (in range 0-1) while applying color,
            # see article "YUV-Farbmodell" in German Wikipedia 
            cy = y[c]
            b = cy + (ufac * cu * color)
            r = cy + (vfac * cv * color)
            g = (cy - 0.299 * r - 0.114 * b) / 0.587
            # Apply picture and brightness, then restrict RGB to 0-1 range
            r = clamp(r * picture + brightness)
            g = clamp(g * picture + brightness)
            b = clamp(b * picture + brightness)
            # linearize the RGB value, i.e. remove gamma correction
            r = linearize_ntsc(r)
            g = linearize_ntsc(g)
            b = linearize_ntsc(b)
            # go from linearized SMPTE-C to CIE1931 XYZ
            xl =  zz[0] * r + zz[1] * g + zz[2] * b
            yl =  zz[3] * r + zz[4] * g + zz[5] * b
            zl =  zz[6] * r + zz[7] * g + zz[8] * b
            # Calculate L*a*b* values from CIE 1931
            fx = _fx(xl)
            fy = _fy(yl)
            fz = _fz(zl)
            l2 = 116. * fy - 16
            a2 = 500. * (fx - fy)
            b2 = 200. * (fy - fz)
            # Apply the CIEDE2000 formula for color difference.
            # Yes, it's pretty complex.
            c2 = math.sqrt(sqr(a2) + sqr(b2))
            l1 = l_star[c]
            a1 = a_star[c]
            b1 = b_star[c]
            c1 = c_star[c]
            l_bar_prime = (l1 + l2) / 2.
            c_bar = (c1 + c2) / 2.
            pow_cb_7 = pow(c_bar, 7)
            g_plus_1 = 1 + (1 - math.sqrt(pow_cb_7 / (pow_cb_7 + pow_25_7))) / 2.
            a1_prime = a1 * g_plus_1
            a2_prime = a2 * g_plus_1
            c1_prime = math.sqrt(sqr(a1_prime) + sqr(b1))
            c2_prime = math.sqrt(sqr(a2_prime) + sqr(b2))
            c_bar_prime = (c1_prime + c2_prime) / 2.
            h1_prime = darctan2(b1, a1_prime)
            if h1_prime < 0:
                h1_prime += 360
            h2_prime = darctan2(b2, a2_prime)
            if h2_prime < 0:
                h2_prime += 360
            if abs(h1_prime - h2_prime) > 180.:
                h_bar_prime = (h1_prime + h2_prime + 360) / 2.
            else:
                h_bar_prime = (h1_prime + h2_prime) / 2.
            t = 1 - 0.17 * dcos(h_bar_prime - 30)
            t = t + 0.24 * dcos(2 * h_bar_prime) + 0.32 * dcos(3 * h_bar_prime + 6)
            t = t - 0.20 * dcos(4 * h_bar_prime - 63)
            if abs(h2_prime - h1_prime) <= 180:
                delta_hs_prime = h2_prime - h1_prime
            elif h2_prime <= h1_prime:
                delta_hs_prime = h2_prime - h1_prime + 360
            else:
                delta_hs_prime = h2_prime - h1_prime - 360
            delta_l_prime = l2 - l1
            delta_c_prime = c2_prime - c1_prime
            delta_h_prime = 2 * math.sqrt(c1_prime * c2_prime) * dsin(delta_hs_prime / 2)
            sqr_lbp_minus_50 = sqr(l_bar_prime - 50)
            sl = 1 + 0.015 * sqr_lbp_minus_50 / math.sqrt(20 + sqr_lbp_minus_50)
            sc = 1 + 0.045 * c_bar_prime
            sh = 1 + 0.015 * c_bar_prime * t
            delta_theta_times_2 = 60 * math.exp(-sqr((h_bar_prime - 275) / 25))
            pow_cbp_7 = pow(c_bar_prime, 7)
            rc = 2 * math.sqrt(pow_cbp_7 / (pow_cbp_7 + pow_25_7))
            rt = -rc * dsin(delta_theta_times_2)
            delta_l_prime /= sl
            delta_c_prime /= sc
            delta_h_prime /= sh
            myerr = sqr(delta_l_prime) + sqr(delta_c_prime) + sqr(delta_h_prime) + rt * delta_c_prime * delta_h_prime
            delta_e[c] = math.sqrt(myerr) # This is the actual CIEDE2000 value 
            err += myerr # We use the squared value here
            if err > MINERR:
                # Save time, skipping if err already too high */
                break
            # go from CIE1931 XYZ to linearized sRGB
            xc[c] = xyz_to_x(xl, yl, zl)
            yc[c] = xyz_to_y(xl, yl, zl)
            ycap_c[c] = yl
            r = a[0] * xl + a[1] * yl + a[2] * zl
            g = a[3] * xl + a[4] * yl + a[5] * zl
            b = a[6] * xl + a[7] * yl + a[8] * zl
            # do gamma correction
            r = gammaize_srgb(r)
            g = gammaize_srgb(g)
            b = gammaize_srgb(b)
            # restrict to 0-1 range and store final values 
            rd[c] = limitprod(r, 256)
            gr[c] = limitprod(g, 256)
            bl[c] = limitprod(b, 256)
        if err < MINERR:
            MINERR = err # ensure we don't go lower
            if attempt > MAXTRY:
                MAXTRY = attempt
            attempt = 0
            err = math.sqrt(err/16.) #  Calc root of mean squared error 
            print_result(brightness=brightness, hue=hue, color=color, picture=picture,
                         err=err, MAXTRY=MAXTRY,
                         xc=xc, yc=yc, ycap_c=ycap_c, rd=rd, gr=gr, bl=bl,
                         riigs=riigs, giigs=giigs, biigs=biigs, 
                         xiigs=xiigs, yiigs=yiigs, y_cap_iigs=y_cap_iigs, delta_e=delta_e)
        else:
            print("No improvement for attempt %d...  MinErr: %.4f (%.8f, %.8f, %.8f, %.8f)" % (attempt, MINERR, brightness, hue, color, picture))
            # no improvement, go back to previous values
            brightness = old_brightness
            picture = old_picture
            color = old_color
            hue = old_hue
            attempt += 1
    
# run main function
main_search(BRIGHTNESS=BRIGHTNESS, HUE=HUE, COLOR=COLOR, PICTURE=PICTURE)

