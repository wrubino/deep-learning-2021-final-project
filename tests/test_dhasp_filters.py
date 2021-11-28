# %% Imports
from toolbox.initialization import *

# %% Initialize the dhasp model.
fs_model = 24e3
dhasp = t.dhasp.DHASP(fs_model)

#%% Show the analysis and control filterbank of the auditory model.
print('Analysis filterbank:')
dhasp.show_filterbank_responses('analysis', show_every=2)
plt.show()
dhasp.show_filterbank_joint_response('analysis')
plt.show()

print('Control filterbank:')
dhasp.show_filterbank_responses('control', show_every=2)
plt.show()
dhasp.show_filterbank_joint_response('control')
plt.show()
