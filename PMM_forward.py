import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import time


# from matplotlib import animation


def rk4_2(func, y0, t0, t1, dt, mu, c, curv_coef, fphix, fphiy, fcurvx, fcurvy, atol, rtol):
    s = ode(func).set_integrator('dopri5', method='bdf', atol=atol, rtol=rtol)
    # 'dopri5': an explicit runge-kutta method of order (4)5,  the local truncation error is on the order of O(h^{5}),
    # while the total accumulated error is on the order of O(h^{4}).

    # 'bdf': Backward Differentiation Formula metohds
    # atol: float or sequence absolute tolerance for solution
    # rtol: float or sequence relative tolerance for solution

    # if type(arg1) == float:
    #    s.set_initial_value(y0, t0).set_f_params(arg1,mu,c,fphix,fphiy)
    # else:
    #    n = len(arg1)
    #    for i in range(n):
    #        s.set_initial_value(y0, t0).set_f_params(arg1[i],mu,c,fphix,fphiy)

    s.set_initial_value(y0, t0).set_f_params(mu, c, curv_coef, fphix, fphiy, fcurvx, fcurvy)

    t = [t0]
    y = [[y0[0], y0[1], y0[2], y0[3]]]

    while s.successful() and s.t < t1:
        # print(s.t+dt, s.integrate(s.t+dt))
        s.t + dt, s.integrate(s.t + dt)
        t.append(s.t)
        y.append(s.y)
    # while s.t < t1:
    #     s.integrate(t1, step=True)
    #     t.append(s.t)
    #     y.append(s.y)

    t = np.array(t)
    y = np.array(y)
    return y, t


def add_annotate(allt, y):
    max_indx = np.argmax(y)  # max value index
    # plt.plot(y,'r-o')
    plt.plot(allt[max_indx], y[max_indx], 'ks')
    show_max = '[' + str(round(allt[max_indx], 1)) + ', ' + str(round(y[max_indx], 1)) + ']'
    plt.annotate(show_max, xytext=(allt[max_indx], y[max_indx]), xy=(allt[max_indx], y[max_indx]),
                 verticalalignment='Top', horizontalalignment='right')


def model4(t, xv, mu, C, curv_coef, fphix, fphiy, fcurvx, fcurvy):
    # xv[posx, posy, vx, vy]
    # arg1 is a tensor(1st or 2nd order) now, arg1[0]= phix(x(t),y(t)), arg1[1] = phiy(x(t),y(t))
    phi = _inclination(xv[0], xv[1], fphix, fphiy)
    curv = _curvature(xv[0], xv[1], fcurvx, fcurvy)
    g = -9.80665  # standard gravitational acceleration

    dxdt = xv[2]
    dydt = xv[3]

    #     if LA.norm(xv[2:4])==0: # there is no friction when velocity is zero
    #         dv0dt = g*np.sin(arg1[0])
    #         dv1dt = g*np.sin(arg1[1])
    #     else:
    #         #dv0dt = g*np.sin(arg1[0]) - mu*np.cos(arg1[0])*g - C*xv[2]/LA.norm(xv[2:4])
    #         #dv1dt = g*np.sin(arg1[1]) - mu*np.cos(arg1[1])*g - C*xv[3]/LA.norm(xv[2:4])
    #         dv0dt = g*np.sin(arg1[0]) - mu*np.cos(arg1[0])*LA.norm(g)*xv[2]/LA.norm(xv[2]) - C*xv[2]*LA.norm(xv[2:4])
    #         dv1dt = g*np.sin(arg1[1]) - mu*np.cos(arg1[1])*LA.norm(g)*xv[3]/LA.norm(xv[3]) - C*xv[3]*LA.norm(xv[2:4])

    if LA.norm(xv[2]) != 0 and LA.norm(xv[3]) != 0:
        dv0dt = (g * np.sin(phi[0])
                 - mu * (np.cos(phi[0]) * LA.norm(g) + curv_coef * curv[0] * np.abs(xv[2]) * LA.norm(xv[2:4])) * xv[
                     2] / LA.norm(xv[2])
                 - C * xv[2] * LA.norm(xv[2:4]))
        dv1dt = (g * np.sin(phi[1])
                 - mu * (np.cos(phi[1]) * LA.norm(g) + curv_coef * curv[1] * np.abs(xv[3]) * LA.norm(xv[2:4])) * xv[
                     3] / LA.norm(xv[3])
                 - C * xv[3] * LA.norm(xv[2:4]))

    elif LA.norm(xv[2]) == 0 and LA.norm(xv[3]) != 0:
        dv0dt = g * np.sin(phi[0])
        dv1dt = (g * np.sin(phi[1])
                 - mu * (np.cos(phi[1]) * LA.norm(g) + curv_coef * curv[1] * np.abs(xv[3]) * LA.norm(xv[2:4])) * xv[
                     3] / LA.norm(xv[3])
                 - C * xv[3] * LA.norm(xv[2:4]))

    elif LA.norm(xv[2]) != 0 and LA.norm(xv[3]) == 0:
        dv0dt = (g * np.sin(phi[0])
                 - mu * (np.cos(phi[0]) * LA.norm(g) + curv_coef * curv[0] * np.abs(xv[2]) * LA.norm(xv[2:4])) * xv[
                     2] / LA.norm(xv[2])
                 - C * xv[2] * LA.norm(xv[2:4]))
        dv1dt = g * np.sin(phi[1])

    else:
        dv0dt = g * np.sin(phi[0])
        dv1dt = g * np.sin(phi[1])

    # the last term is scaled by the norm of local velocity
    dxvdt = [dxdt, dydt, dv0dt, dv1dt]

    return dxvdt


def _inclination(x, y, fphix, fphiy):
    argx = fphix(y, x)
    argy = fphiy(y, x)
    arg = [argx, argy]  # radian

    return arg


def _curvature(x, y, fcurvx, fcurvy):
    argx = fcurvx(y, x)
    argy = fcurvy(y, x)
    arg = [argx, argy]  # Rad

    return arg


def _curvature_base(x, y, topo):
    # arg1 is the dimensional slope from the original topography
    arg1 = np.gradient(topo, x, y)
    gradx = np.gradient(arg1[0], x, y)
    gradxx = gradx[0]
    gradxy = gradx[1]

    grady = np.gradient(arg1[1], x, y)
    # gradyx = grady[0] #the same as gradxy
    gradyy = grady[1]

    # the mean curvature
    h = ((1 + arg1[0] ** 2) * gradyy - 2 * arg1[0] * arg1[1] * gradxy + (1 + arg1[1] ** 2) * gradxx) / (
            2 * (1 + arg1[0] ** 2 + arg1[1] ** 2) ** 1.5)
    # Gaussian curvature
    k = (gradxx * gradyy - gradxy ** 2) / (1 + arg1[0] ** 2 + arg1[1] ** 2) ** 2

    curv1 = h + np.sqrt(h ** 2 - k)  # the maximal curvature
    # curv2 = h - np.sqrt(h ** 2 - k)  # the minimal curvature

    curvx = gradxx / ((1 + arg1[0] ** 2) ** 1.5)
    curvy = gradyy / ((1 + arg1[1] ** 2) ** 1.5)
    curv = [curvx, curvy]  # curvature in 2 dimensions
    return curv1, curv


def set_topo(x_ini, y_ini, cell, mu, topotype=''):
    if topotype == '0' or topotype == '1':  # incline plane
        x = np.arange(x_ini, x_ini + n * cell, cell)
        y = np.arange(y_ini + m * cell - 6000, y_ini + m * cell, cell)
        mu = mu
        b = 800 - mu * 2400
        z_mu = [mu * (e - y_ini) + b for e in y]  # effective friction angle

        if topotype == '0':  # sythetic topo A
            za = [1 / 3 * (e - y_ini) for e in y]  # incline plane
        else:  # sythetic topo B
            za = [1 / 7200 * (e - y_ini) ** 2 for e in y]  # parabolic sythetic topo

        for i in range(len(y)):
            if z_mu[i] < 0:
                z_mu[i] = 0
            if y[i] < y_ini:
                za[i] = 0

        y_stop = (-b / mu + y_ini)
        ###################################### plot topo profile ##########################################
        # fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(111)
        # ax.plot(y, za, label="Sythetic topography profile")
        # ax.plot(y, z_mu, '--', lw=1, label="Effectice friction angle")
        # ax.set_xlabel("y [m]")
        # ax.set_ylabel("z [m]")
        # ax.set_title("2D topography")
        # ax.legend(loc=2)
        #
        #
        # ax.scatter(y_stop, 0)
        # plt.show()
        #
        # fig.savefig('pics/sythetic_topo/0.15.png', bbox_inches='tight')

        E1 = za * np.ones([len(x), 1])  # sythetic topo
        topo = E1.copy()

    else:
        x = np.arange(x_ini, x_ini + n * cell, cell)
        y = np.arange(y_ini, y_ini + m * cell, cell)  # 2D sample topography a(x,y)
        topo = dhm.copy()
        y_stop = None
    return x, y, topo, y_stop


def PMM_forward(x_ini, y_ini, cell, location, mu, c, curv_coef, topotype='', plot_coef=False):
    global x_norm, v_norm
    dhm = np.loadtxt('dhm_test.txt', skiprows=6)
    [n, m] = dhm.shape

    x, y, topo, y_stop = set_topo(x_ini, y_ini, cell, mu, topotype)

    x0 = location[0]
    y0 = location[1]
    xv0 = [x0, y0, 0, 0]
    t0 = 0
    t1 = 200
    dt = 0.1

    # mu = 0.15 # dry friction coef
    c = c  # 0.01~0.0001

    atol = 1e-6
    rtol = 1e-6

    # get the dimensional slope from the original topography
    arg1 = np.gradient(topo, x, y)
    arg1x = np.arctan(arg1[0], dtype=np.float)  # .shape = [352,605] in Rad
    arg1y = np.arctan(arg1[1], dtype=np.float)  # .shape = [352,605]
    # interpolate of slope
    fphix = interpolate.interp2d(y, x, arg1x, kind='cubic')
    fphiy = interpolate.interp2d(y, x, arg1y, kind='cubic')

    # get the dimensional curvature from the original topography
    curvmax, curv = _curvature_base(x, y, topo)

    curvx = curv[0]
    curvy = curv[1]

    fcurvx = interpolate.interp2d(y, x, curvx, kind='cubic')
    fcurvy = interpolate.interp2d(y, x, curvy, kind='cubic')

    curv_coef = curv_coef

    if topotype == '0' or topotype == '1':
        if curv_coef == 0:
            xv, t = rk4_2(model4, xv0, t0, t1, dt, mu, c, 0, fphix, fphiy, fcurvx, fcurvy, atol,
                          rtol)  # with curvature effect

            v_norm = []
            for i in range(len(xv)):
                v_norm.append(LA.norm(xv[i][2:4]))  # magnitude of velocity
            x_norm = []
            for i in range(len(xv)):
                x_norm.append(LA.norm(xv[0][0:2] - xv[i][0:2]))  # magnitude of displacement
                # x_norm.append(LA.norm(xv[i][0:2]))
            #########################################################################################################
            # _plotvx("pics/sythetic_topo/vx1_0_test.png", "2D velocity & displacement, without curvature effect",
            #         "time t",
            #         "v(t)[m/s]", "x(t)[m]", t, v_norm, x_norm, y0 - y_stop, stop=True)

        else:
            xv, t = rk4_2(model4, xv0, t0, t1, dt, mu, c, 1, fphix, fphiy, fcurvx, fcurvy, atol,
                          rtol)  # without curvature effect

            v_norm = []
            for i in range(len(xv)):
                v_norm.append(LA.norm(xv[i][2:4]))  # magnitude of velocity
            x_norm = []
            for i in range(len(xv)):
                x_norm.append(LA.norm(xv[0][0:2] - xv[i][0:2]))  # magnitude of displacement
                # x_norm.append(LA.norm(xv[i][0:2]))
            #########################################################################################################
            # _plotvx("pics/sythetic_topo/vx1_1_test.png", "2D velocity & displacement, with curvature effect", "time t",
            #         'v(t)[m/s]', "x(t)[m]", t, v_norm, x_norm, y0 - y_stop, stop=True)

    else:
        xv, t = rk4_2(model4, xv0, t0, t1, dt, mu, c, 0, fphix, fphiy, fcurvx, fcurvy, atol,
                      rtol)  # with curvature effect

        v_norm = []
        for i in range(len(xv)):
            v_norm.append(LA.norm(xv[i][2:4]))  # magnitude of velocity
        x_norm = []
        for i in range(len(xv)):
            x_norm.append(LA.norm(xv[0][0:2] - xv[i][0:2]))  # magnitude of displacement
            # x_norm.append(LA.norm(xv[i][0:2]))
        #########################################################################################################
        #
        # _plotvx("pics/sythetic_topo/vx1_0_test.png", "2D velocity & displacement, without curvature effect",
        #         "time t",
        #         "v(t)[m/s]", "x(t)[m]", t, v_norm, x_norm, y_stop='None', stop=False)

    # interpolate of altitudea
    f = interpolate.interp2d(y, x, topo, kind='cubic')

    points = list(zip(list(np.copy(xv[:, 1])), list(np.copy(xv[:, 0]))))
    # z = []
    z = [float(f(*p)) for p in points]

    #########################################################################################################

    # fig = plt.figure(figsize=(16, 12))
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    # X, Y = np.meshgrid(x, y, indexing="ij")
    # im = ax.plot_surface(X, Y, topo, cmap='viridis')
    # # im = ax.contour(X, Y, dhm, cmap='viridis')
    # plt.colorbar(im, ax=ax, shrink=.5, pad=.02)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    #
    # ax.plot(xv[:, 0], xv[:, 1], z)
    #
    # # z_xv = f(xv[0][1], xv[0][0])
    # ax.scatter(xv[0][0], xv[0][1], z[0], s=100, c='r')
    #
    # ax.set_zlabel("Altitude [m]")
    # if topotype == '0' or topotype == '1':
    #     ax.set_title("Trajactory on Sythetic Topography")
    #     ax.plot(x, y_stop * np.ones([len(x), 1]))
    # else:
    #     ax.set_title("Trajactory on Sample Topography")
    #     pass
    # # fig.savefig('pics/sythetic_topo/trajactory1_0.15.png', bbox_inches='tight')
    # plt.show()
    if plot_coef == False:
        pass
    else:
        _plot_inclination(x, y, topo)
        _plot_curvature(x, y, topo)
    ################################  visualize inlination  ################################################
    # incline = []
    # [a, b] = topo.shape
    # incline = _inclination(x, y, fphix, fphiy)
    # phi = list(zip(list(np.copy(incline[0].reshape(a * b))), list(np.copy(incline[1].reshape(a * b)))))
    #
    # phinorm = []
    # phinorm.append(LA.norm(phi, axis=1))
    #
    # phi_norm = phinorm[0].reshape(a, b)
    #
    # norm = matplotlib.colors.Normalize(vmin=phi_norm.min().min(), vmax=phi_norm.max().max())
    #
    #
    # fig = plt.figure(figsize=(16, 12))
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    # X, Y = np.meshgrid(x, y, indexing="ij")
    # ax.plot_surface(X, Y, topo, facecolors=plt.cm.plasma(norm(phi_norm)), shade=False)
    # mm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    # mm.set_array([])
    # # plt.colorbar(m)
    #
    # # plt.colorbar(m)
    # # im = ax.contour(X, Y, dhm, cmap='viridis')
    # plt.colorbar(mm, ax=ax, shrink=.5, pad=.02)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # ax.set_title("inclination")
    # plt.show()
    #
    # fig.savefig('pics/sythetic_topo/inclination.png', bbox_inches='tight')

    ########################### visualize curvature ###########################################################

    # curvmax, curv = _curvature_base(x, y, arg1)
    #
    # fig = plt.figure(figsize=(16, 12))
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    # X, Y = np.meshgrid(x, y, indexing="ij")
    # norm = matplotlib.colors.Normalize(vmin=curvmax.min().min(), vmax=curvmax.max().max())
    #
    # ax.plot_surface(X, Y, topo, facecolors=plt.cm.plasma(norm(curvmax)), shade=False)
    # mm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    # mm.set_array([])
    # plt.colorbar(mm, ax=ax, shrink=.5, pad=.02)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    #
    # ax.set_title("the maximal curvature")
    # plt.show()
    #
    # fig.savefig('pics/sythetic_topo/curvature_max.png', bbox_inches='tight')
    return t, xv, z, v_norm, x_norm


##################################plotting funtions######################################################################

def _plotvx(out, title, xlabel, ylabel1, ylabel2, allt, y1, y2, y_stop, stop=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(allt, y1, 'b', label="Velocity")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    add_annotate(allt, y1)
    # ax1.legend(loc=4)
    ax2 = ax1.twinx()
    line2 = ax2.plot(allt, y2, 'r', label="Displacement")
    ax2.set_ylabel(ylabel2)
    ax2.set_title(title)
    add_annotate(allt, y2)
    if y_stop == 'None':
        line3 = []
        pass
    else:
        if stop:
            line3 = ax2.plot(allt, y_stop * np.ones([len(allt), 1]), 'g--', label='friction_eff stop location')
            show = '[' + str(round(allt[-1], 1)) + ', ' + str(round(y_stop, 1)) + ']'
            ax2.annotate(show, xytext=(allt[-1], y_stop), xy=(allt[-1], y_stop),
                         verticalalignment='Top', horizontalalignment='right')
        else:
            line3 = []
            pass

    lines = []
    lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=2)
    plt.grid(True)
    plt.savefig(out, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False)
    plt.show()


def plot_trajactory(x_ini, y_ini, cell, mu, xv, z, topotype='', stop=True):
    x, y, topo, y_stop = set_topo(x_ini, y_ini, cell, mu, topotype)

    points = list(zip(list(np.copy(xv[:, 1])), list(np.copy(xv[:, 0]))))
    fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x, y, indexing="ij")
    im = ax.plot_surface(X, Y, topo, cmap='viridis')
    # im = ax.contour(X, Y, dhm, cmap='viridis')
    plt.colorbar(im, ax=ax, shrink=.5, pad=.02)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.plot(xv[:, 0], xv[:, 1], z)

    # z_xv = f(xv[0][1], xv[0][0])
    ax.scatter(xv[0][0], xv[0][1], z[0], s=100, c='r')

    ax.set_zlabel("Altitude [m]")
    if stop:
        if topotype == '0' or topotype == '1':
            ax.set_title("Trajactory on Sythetic Topography")
            ax.plot(x, y_stop * np.ones([len(x), 1]))
        else:
            ax.set_title("Trajactory on Sample Topography")
            pass
    else:
        pass

    # fig.savefig('pics/sythetic_topo/trajactory1_0.15.png', bbox_inches='tight')
    # plt.show()


def _plot_inclination(x, y, topo):
    ###############################  visualize inlination  ################################################
    [a, b] = topo.shape
    # incline = _inclination(x, y, fphix, fphiy)
    arg1 = np.gradient(topo, x, y)
    arg1x = np.arctan(arg1[0], dtype=np.float)  # .shape = [352,605] in Rad
    arg1y = np.arctan(arg1[1], dtype=np.float)
    phi = list(zip(list(np.copy(arg1x.reshape(a * b))), list(np.copy(arg1y.reshape(a * b)))))

    phinorm = []
    phinorm.append(LA.norm(phi, axis=1))

    phi_norm = phinorm[0].reshape(a, b)

    norm = matplotlib.colors.Normalize(vmin=phi_norm.min().min(), vmax=phi_norm.max().max())

    fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.plot_surface(X, Y, topo, facecolors=plt.cm.plasma(norm(phi_norm)), shade=False)
    mm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    mm.set_array([])
    # plt.colorbar(m)

    # plt.colorbar(m)
    # im = ax.contour(X, Y, dhm, cmap='viridis')
    plt.colorbar(mm, ax=ax, shrink=.5, pad=.02)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("inclination")
    plt.show()

    fig.savefig('results/inclination.png', bbox_inches='tight')

    ##########################


def _plot_curvature(x, y, topo):
    curvmax, curv = _curvature_base(x, y, topo)

    fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x, y, indexing="ij")
    norm = matplotlib.colors.Normalize(vmin=curvmax.min().min(), vmax=curvmax.max().max())

    ax.plot_surface(X, Y, topo, facecolors=plt.cm.plasma(norm(curvmax)), shade=False)
    mm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    mm.set_array([])
    plt.colorbar(mm, ax=ax, shrink=.5, pad=.02)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title("the maximal curvature")
    plt.show()

    fig.savefig('results/curvature_max.png', bbox_inches='tight')


################################## sampling funtions ######################################################################

def sample_Gaussian(x_mean, y_mean, gaussian_sample_coef, N, plot=True):
    mean = [x_mean, y_mean]
    # cov = gaussian_sample_coef * [[1, 0], [0, 1]]
    cov = [[1 * gaussian_sample_coef, 0 * gaussian_sample_coef], [0 * gaussian_sample_coef, 1 * gaussian_sample_coef]]
    x_gaussian_rand, y_gaussian_rand = np.random.multivariate_normal(mean, cov, N).T
    if plot == True:
        plt.subplot(121)
        plt.scatter(x_gaussian_rand, y_gaussian_rand, marker='x')
        plt.scatter(x_mean, y_mean, c='r', marker='o')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.title('Sampling from a normal distribution')
        plt.subplot(222)
        countx, binsx, ignoredx = plt.hist(x_gaussian_rand, bins='auto', density=True)
        plt.xlabel('x[m]')
        plt.ylabel('Probability')
        plt.title('Histogram of x_initial')
        plt.subplot(224)
        county, binsy, ignoredy = plt.hist(y_gaussian_rand, bins='auto', density=True)
        plt.xlabel('y[m]')
        plt.ylabel('Probability')
        plt.title('Histogram of y_initial')
        plt.tight_layout()
        plt.show()
    else:
        pass
    sample_points = list(zip(x_gaussian_rand.copy(), y_gaussian_rand.copy()))
    return sample_points


def sample_uniform(x_mean, y_mean, uni_sample_coef, N, plot=True):
    # uni_sample_coef is the half of the range of sampling
    x_min = x_mean - uni_sample_coef
    x_max = x_mean + uni_sample_coef
    y_min = y_mean - uni_sample_coef
    y_max = y_mean + uni_sample_coef
    x_uni_rand = np.random.uniform(x_min, x_max, N)
    y_uni_rand = np.random.uniform(y_min, y_max, N)
    if plot == True:
        plt.subplot(121)
        plt.scatter(x_uni_rand, y_uni_rand, marker='x')
        plt.scatter(x_mean, y_mean, c='r', marker='o')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.title('Sampling from a uniform distribution')
        plt.subplot(222)
        countx, binsx, ignoredx = plt.hist(x_uni_rand, bins='auto', density=True)
        plt.xlabel('x[m]')
        plt.ylabel('Probability')
        plt.title('Histogram of x_initial')
        plt.subplot(224)
        county, binsy, ignoredy = plt.hist(y_uni_rand, bins='auto', density=True)
        plt.xlabel('y[m]')
        plt.ylabel('Probability')
        plt.title('Histogram of y_initial')
        plt.tight_layout()
        plt.show()
    else:
        pass
    sample_points = list(zip(x_uni_rand.copy(), y_uni_rand.copy()))
    return sample_points


start = time.clock()

dhm = np.loadtxt('dhm_test.txt', skiprows=6)
[n, m] = dhm.shape
x_ini = 784479.
y_ini = 188093.
cell = 4
location_ini = [x_ini + 0.6 * n * cell, y_ini + 1 * m * cell]
c = 0.000
mu = 0.15
curv_coef = 0
x, y, topo, y_stop = set_topo(x_ini, y_ini, cell, mu, topotype='1')
t, xv_center, z, v_norm, x_norm = PMM_forward(x_ini, y_ini, cell, location_ini, mu, c, curv_coef, '1', False)
#
_plotvx("results/vx1_0_test.png", "2D velocity & displacement, without curvature effect",
        "time t",
        "v(t)[m/s]", "x(t)[m]", t, v_norm, x_norm, xv_center[0][1] - y_stop, stop=True)

# _plotvx("pics/sythetic_topo/vx1_0_test.png", "2D velocity & displacement, with curvature effect",
#                 "time t",
#                 "v(t)[m/s]", "x(t)[m]", t, v_norm, x_norm, xv_center[0][1] - y_stop, stop=True)
#


N = 20
points_uniform = sample_uniform(location_ini[0], location_ini[1], 200, N, plot=True)
points_gaussian = sample_Gaussian(location_ini[0], location_ini[1], 800, N, plot=True)

x_end = []
y_end = []
plt.figure(figsize=(16, 12))

for i in range(N):
    location = points_uniform[i]
    t, xv, z, v_norm, x_norm = PMM_forward(x_ini, y_ini, cell, location, mu, c, curv_coef, '0', False)
    x_end.append(xv[-1, 0])
    y_end.append(xv[-1, 1])
    # plot_trajactory(x_ini, y_ini, cell, mu, xv, z, topotype='1', stop=False)
    # _plotvx("pics/sythetic_topo/vx1_1_test.png", "2D velocity & displacement", "time t",
    #         'v(t)[m/s]', "x(t)[m]", t, v_norm, x_norm, 'None', stop=False)

    plt.plot(t, np.abs(xv[:, 3]), lw=1, ls='--')
    # plt.plot(xv[:, 0], xv[:, 1])
    # plt.scatter(xv[0][0], xv[0][1], c='b',marker='x')
    # plt.scatter(xv[-1][0], xv[-1][1], c='y',marker = 'x')

    # # plt.show()

t, xv_center, z, v_norm, x_norm = PMM_forward(x_ini, y_ini, cell, location_ini, mu, c, curv_coef, '0', False)
plt.plot(t, np.abs(xv_center[:, 3]), lw=2, c='r')
plt.xlabel('t[s]')
# plt.ylabel('y[m]')
# plt.title("Displacement")
plt.ylabel('v[m/s]')
plt.title("Velocity")

# plt.scatter(location_ini[0], location_ini[1], c='r', marker='o')
# plt.plot(xv_center[:, 0], xv_center[:, 1],c='r',lw=2)
# plt.axis([x[0], x[-1], y[0], y[-1]])
# plt.xlabel('x[m]')
# plt.ylabel('y[m]')
# plt.title("Top view of the trajactory ")

#
plt.show()
#
# elapsed = (time.clock() - start)
# print("Time used:", elapsed)
#
# points_gaussian_test = sample_Gaussian(location_ini[0], location_ini[1], 800, 500, plot=True)
# points_uni_test = sample_uniform(location_ini[0], location_ini[1], 200, 500, plot=True)
