import xtrack as xt

env_ref = xt.load('../lattices/z/_temp/env_no_bpms.json')
env_test = xt.load('../lattices/z/_temp/env_no_bpms_merged_quads.json')

line_ref = env_ref.fccee_p_ring
line_test = env_test.fccee_p_ring

tt_ref = line_ref.get_table(attr=True)
tt_test = line_test.get_table(attr=True)

tw_ref = line_ref.twiss4d()
tw_test = line_test.twiss4d()

k1_sum_ref = tt_ref.k1l.cumsum()
k1_sum_test = tt_test.k1l.cumsum()

k1_interp = np.interp(tt_ref.s, tt_test.s, k1_sum_test)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(tt_ref.s, k1_sum_ref, label='ref')
plt.plot(tt_test.s, k1_sum_test, label='test')
plt.plot(tt_ref.s, k1_interp, label='interp')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('Cumulative k1L')
plt.title('Cumulative k1L along the ring')
plt.grid()

# Plot the difference
plt.figure()
plt.plot(tt_ref.s, k1_sum_ref - k1_interp, label='ref - interp')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('Difference in cumulative k1L')
plt.title('Difference in cumulative k1L along the ring')
plt.grid()
plt.show()
