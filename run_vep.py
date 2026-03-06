from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle
import mne

# set to True when doing actual experiment 
cyton_in = False

lsl_out = False
width = 1536
height = 864
aspect_ratio = width/height
refresh_rate = 60.02
n_per_class = 2
recording_mode = True

# duration
stim_duration = 0.3
interval_duration = 0.5

# modify!!!
session = 1
subject = 1
trial_person = '1'
is_test = True
run = 1 # Run number, it is used as the random seed for the trial sequence generation

# for trial data
num_images = 100
num_targets = 10
runs_array = []

save_dir = f'data/cyton8_p300-class_{stim_duration}s/sub-{subject:02d}/ses-{session:02d}/person-{trial_person}' # Directory to save data to
save_file_eeg = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_timestamp = save_dir + f'timestamp_{n_per_class}-per-class_run-{run}.npy'
save_file_metadata = save_dir + f'metadata_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'
model_file_path = 'cache/FBTRCA_model.pkl'

import string
import numpy as np
import psychopy.visual
import psychopy.event
from psychopy import core

keyboard = keyboard.Keyboard()
window = visual.Window(
        size = [width,height],
        checkTiming = True,
        allowGUI = False,
        fullscr = True,
        useRetina = False,
    )

def create_photosensor_dot(size=2/8*0.7):
    width, height = window.size
    ratio = width/height
    return visual.Rect(win=window, units="norm", width=size, height=size * ratio, 
                       fillColor='white', lineWidth = 0, pos = [1 - size/2, -1 - size/8]
    )

def select_indices():
    choices = set()
    while len(choices) < num_targets:
        temp = random.randint(1, num_images-1)
        to_add = True
        for num in choices:
            if abs(temp - num) < 5:
                to_add = False
                break
        if to_add:
            choices.add(temp)
    return choices

def select_targets(person):
    folder = 'target\\'+person
    image_array = np.array([os.path.join(folder, f) for f in os.listdir(folder)])
    selected_images = np.random.choice(image_array, size=num_targets, replace=False)
    return selected_images

def select_nontarget():
    folder = 'nontarget\\real'
    image_array = np.array([os.path.join(folder, f) for f in os.listdir(folder)])
    selected_images = np.random.choice(image_array, size=num_images, replace=False)
    return selected_images

# creates trial sequence
image_array = select_nontarget()
targets = select_targets(trial_person)
indices = select_indices()
count = 0

if not is_test:
    for i in indices:
        image_array[i] = targets[count]
        count += 1

if cyton_in:
    import glob, sys, time, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue
    sampling_rate = 250
    CYTON_BOARD_ID = 0 # 0 if no daisy 2 if use daisy board, 6 if using daisy+wifi shield
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2' # Reads from analog pins A5(D11), A6(D12) and if no 
                        # wifi shield is present, then A7(D13) as well.
    def find_openbci_port():
        """Finds the port to which the Cyton Dongle is connected to."""
        # Find serial port names per OS
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
            exit()
        else:
            return openbci_port
        
    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    res_query = board.config_board('/0')
    print(res_query)
    res_query = board.config_board('//')
    print(res_query)
    res_query = board.config_board(ANALOGUE_MODE)
    print(res_query)
    board.start_stream(45000)
    stop_event = Event()
    
    def get_data(queue_in, lsl_out=False):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)
    
    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None

# for cyton
num_frames = np.round(stim_duration * refresh_rate).astype(int)  # total number of frames per trial
num_wait_frames = np.round(interval_duration * refresh_rate).astype(int) # total number of frames for the wait interval
frame_indices = np.arange(num_frames)  # frame indices for the trial
eeg = np.zeros((8, 0))
aux = np.zeros((3, 0))
timestamp = np.zeros((0))
eeg_trials = []
aux_trials = []
trial_ends = []
skip_count = 0 # Number of trials to skip due to frame miss in those trials

photosensor_dot = create_photosensor_dot()
photosensor_dot.color = np.array([-1, -1, -1])
photosensor_dot.draw()

if recording_mode:
    for i_trial in range(num_images):

        # set the trial image and trial number text
        trial_text = visual.TextStim(window, text=f'Trial {i_trial+1}/{num_images}', pos=(0, -1+0.07), color='white', units='norm', height=0.07)
        image_stim = visual.ImageStim(win=window, image=str(image_array[i_trial]), units="pix", size=(512, 512))
        photosensor_dot.color = np.array([1, 1, 1])

        # set the marker value fr the current trial onlt if cyton_in is True
        if cyton_in:
            if str(image_array[i_trial]) in targets:
                marker_val = 1
            else:
                marker_val = 0
            window.callOnFlip(board.insert_marker, marker_val)

        for i_frame in range(num_frames):
            next_flip = window.getFutureFlipTime()

            keys = keyboard.getKeys()
            if 'escape' in keys:
                    if cyton_in:
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(save_file_eeg, eeg)
                        np.save(save_file_aux, aux)
                        # np.save(save_file_timestamp, timestamp)
                        # np.save(save_file_eeg_trials, eeg_trials)
                        # np.save(save_file_aux_trials, aux_trials)
                        stop_event.set()
                        board.stop_stream()
                        board.release_session()
                    core.quit()

            trial_text.draw()
            image_stim.draw()
            photosensor_dot.draw()
            
            if core.getTime() > next_flip and i_frame != 0:
                print('Missed frame')
            window.flip()

        photosensor_dot.color = np.array([-1, -1, -1])

        for i_int in range(num_wait_frames):
            trial_text.draw() 
            photosensor_dot.draw()
            window.flip() 

        if cyton_in:
            while not queue_in.empty(): # Collect all data from the queue
                eeg_in, aux_in, timestamp_in = queue_in.get()
                print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                eeg = np.concatenate((eeg, eeg_in), axis=1)
                aux = np.concatenate((aux, aux_in), axis=1)
                timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
            # photo_trigger = (aux[1] > 20).astype(int)

    if cyton_in:
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_file_eeg, eeg)
        np.save(save_file_aux, aux)
        board.stop_stream()
        board.release_session()
    
    print('Done.')