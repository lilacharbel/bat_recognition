import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = '../../data/training_data'

samples = []
for filename in os.listdir(data_dir):
    samples.append(len(os.listdir(os.path.join(data_dir, filename))))
samples = np.array(samples)

# plot and save histogram
plt.figure(figsize=(17, 10))
plt.bar(np.arange(96), samples, align='center')
plt.title("Number of samples for each Bat", fontsize=24)
plt.xlabel("Bat", fontsize=20)
plt.xticks(np.arange(96), rotation=90)
plt.ylabel("Number of samples", fontsize=20)
plt.savefig('../../data/figures/data_hist.png')
plt.ylim([0, 200])
plt.savefig('../../data/figures/data_hist_zoom.png')


# plt.figure(figsize=(15, 10))
# plt.bar(np.arange(96), samples/samples.max(), align='center')
# plt.title("Number of samples for each Bat")
# plt.savefig('../../data/figures/data_hist_max.png')
# plt.ylim([0, 0.2])
# plt.savefig('../../data/figures/data_hist_max_zoom.png')
#
# plt.figure(figsize=(15, 10))
# plt.bar(np.arange(96), samples/samples.sum(), align='center')
# plt.title("Number of samples for each Bat")
# plt.savefig('../../data/figures/data_hist_sum.png')
# plt.ylim([0, 0.002])
# plt.savefig('../../data/figures/data_hist_sum_zoom.png')

print('.')