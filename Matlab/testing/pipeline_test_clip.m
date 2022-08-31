%% Path parameters
ieeg_folder = '/Users/erinconrad/Desktop/research/spike_locations/scripts/tools/ieeg-matlab-1.13.2';
ieeg_pw_file = '/Users/erinconrad/Desktop/research/spike_locations/scripts/tools/eri_ieeglogin.bin';
ieeg_login = 'erinconr';

%% Clip parameters
file_name = 'HUP212_phaseII';
times = [100000 100015];
which_reference = 'car';

%% Add paths
% Add path to this codebase
addpath(genpath('./..'))

% Add path to ieeg codebase
addpath(genpath(ieeg_folder))

%% Get patient name
ptnameC = strsplit(file_name,'_');
name = ptnameC{1};

%% Download data from ieeg.org
data = download_ieeg_data(file_name, ieeg_login, ieeg_pw_file, times, 1);
oldLabels = data.chLabels;
old_values = data.values;
fs = data.fs;
nchs = size(old_values,2);

%% Decompose labels
[labels,~,~] = decompose_labels(oldLabels,name);

%% Non intracranial
extra_cranial = find_non_intracranial(labels);

%% Identify bad channels
bad = identify_bad_chs(old_values,fs);
%bad = logical(zeros(nchs,1));

%% Notch Filter
old_values = notch_filter(old_values,fs);

%% Common average reference (include only intra-cranial)
[values,car_labels] = common_average_reference(old_values,~extra_cranial&~bad,labels);

%% Remove extra-cranial and bad
show_values = values(:,~extra_cranial & ~bad);
car_show_labels = car_labels(~extra_cranial & ~bad);

%% Plot the EEG
show_eeg(show_values,fs,car_show_labels)