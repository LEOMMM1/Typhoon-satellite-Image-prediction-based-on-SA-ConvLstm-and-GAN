from torch.utils.data import DataLoader, Dataset
from PIL import Image
import zipfile
import fnmatch
import io
import re
import numpy as np

# load train dataset
class loaded_train_Dataset(Dataset):
    def __init__(self, zip_file, sid, frame_num, time_freq, ps):
        self.zip_file = zipfile.ZipFile(zip_file, 'r')
        self.sid = str(sid)
        self.fnum = frame_num
        self.freq = time_freq
        self.ps = ps

        # show all matched files in the zip file
        filename = self.sid + '_Hour_*_Band_09.png'
        all_files = [name for name in self.zip_file.namelist() if fnmatch.fnmatch(name, filename)]

        # filter files, only keep the files with number between 1 and max_limit
        if self.sid == 'A':
            max_limit = 1865
        else:
            max_limit = 1987

        self.files = []
        for file in all_files:
            match = re.search(r'Hour_(\d+)', file)
            if match:
                num = int(match.group(1))
                if 1 <= num <= max_limit:
                    self.files.append(file)

        # save the loaded data in memory. It will take a long time to load 12 images each time.
        self.allframes = [None] * len(self.files)
        tids = []
        for i in range(len(self.allframes)):
            self.allframes[i] = []
            tids.append(int(re.findall(r'Hour_(\d+)', self.files[i])[0]))

        print(tids[:10])

        self.first_tid = min(tids)
        self.last_tid = max(tids)
        self.tids = sorted(tids)

    def __len__(self):
        return len(self.files) - (self.fnum // 2) * (self.freq + 1) + 1

    def __getitem__(self, idx):
        # show the time point ID of 12 frames, e.g., the time point ID of A_Hour_189 is 189
        tid = self.tids[idx]

        # get 6 continuous frames from the current time point, and take them as the first 6 frames to train
        ids1 = list(range(tid, tid + (self.fnum // 2)))

        # get 6 continuous frames from the current time point, and take them as the last 6 frames to train, with the interval of self.freq.
        # These will be the target frames we want to predict.
        ids2 = list(range(tid + (self.fnum // 2) + self.freq - 1,
                          tid + (self.fnum // 2) + self.freq + (self.fnum // 2) * self.freq - 1, self.freq))
        ids = ids1 + ids2

        frames = []  
        for i, ctid in enumerate(ids):

            if not len(self.allframes[ctid - self.first_tid]):
                with self.zip_file.open(self.sid + '_Hour_' + str(ctid) + '_Band_09.png') as file:
                    frame = np.array(Image.open(io.BytesIO(file.read()))) / 255.0  # normalizatino
                    frame = frame[:, :, np.newaxis]
                    frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])  # CHW->HWC
                    self.allframes[ctid - self.first_tid] = frame
            else:
                frame = self.allframes[ctid - self.first_tid]

            frames.append(frame)

        return ids, frames


# load valid dataset
class loaded_valid_Dataset(Dataset):
    def __init__(self, zip_file, sid, frame_num, time_freq, ps):
        self.zip_file = zipfile.ZipFile(zip_file, 'r')
        self.sid = str(sid)
        self.fnum = frame_num
        self.freq = time_freq
        self.ps = ps

        filename = self.sid + '_Hour_*_Band_09.png'
        all_files = [name for name in self.zip_file.namelist() if fnmatch.fnmatch(name, filename)]

        if self.sid == 'A':
            min_limit = 1865
        else:
            min_limit = 1987
        self.files = []
        for file in all_files:
            match = re.search(r'Hour_(\d+)', file)
            if match:
                num = int(match.group(1))
                if min_limit <= num <= 2208:
                    self.files.append(file)

        self.allframes = [None] * len(self.files)
        tids = []
        for i in range(len(self.allframes)):
            self.allframes[i] = []
            tids.append(int(re.findall(r'Hour_(\d+)', self.files[i])[0]))

        print(tids[:10])


        self.first_tid = min(tids)
        self.last_tid = max(tids)
        self.tids = sorted(tids)

    def __len__(self):
        return len(self.files) - (self.fnum // 2) * (self.freq + 1) + 1

    def __getitem__(self, idx):
        tid = self.tids[idx]
        ids1 = list(range(tid, tid + (self.fnum // 2)))
        ids2 = list(range(tid + (self.fnum // 2) + self.freq - 1,
                          tid + (self.fnum // 2) + self.freq + (self.fnum // 2) * self.freq - 1, self.freq))
        ids = ids1 + ids2

        frames = [] 
        for i, ctid in enumerate(ids):
            if not len(self.allframes[ctid - self.first_tid]):
                with self.zip_file.open(self.sid + '_Hour_' + str(ctid) + '_Band_09.png') as file:
                    frame = np.array(Image.open(io.BytesIO(file.read()))) / 255.0  
                    frame = frame[:, :, np.newaxis]
                    frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])  
                    self.allframes[ctid - self.first_tid] = frame
            else:
                frame = self.allframes[ctid - self.first_tid]

            frames.append(frame)

        return ids, frames


# load test dataset
class loaded_test_Dataset(Dataset):
    def __init__(self, zip_file, sid, frame_num, time_freq, ps):
        self.zip_file = zipfile.ZipFile(zip_file, 'r')
        self.sid = str(sid)
        self.fnum = frame_num
        self.freq = time_freq
        self.ps = ps

        filename = self.sid + '_Hour_*_Band_09.png'
        all_files = [name for name in self.zip_file.namelist() if fnmatch.fnmatch(name, filename)]


        self.files = []
        for file in all_files:
            match = re.search(r'Hour_(\d+)', file)
            if match:
                num = int(match.group(1))
                if 1 <= num:
                    self.files.append(file)


        self.allframes = [None] * len(self.files)
        tids = []
        for i in range(len(self.allframes)):
            self.allframes[i] = []
            tids.append(int(re.findall(r'Hour_(\d+)', self.files[i])[0]))

        print(tids[:10])


        self.first_tid = min(tids)
        self.last_tid = max(tids)
        self.tids = sorted(tids)

    def __len__(self):
        return len(self.files) - (self.fnum // 2) * (self.freq + 1) + 1

    def __getitem__(self, idx):
        tid = self.tids[idx]
        ids1 = list(range(tid, tid + (self.fnum // 2)))
        ids2 = list(range(tid + (self.fnum // 2) + self.freq - 1,
                          tid + (self.fnum // 2) + self.freq + (self.fnum // 2) * self.freq - 1, self.freq))
        ids = ids1 + ids2

        frames = []  
        for i, ctid in enumerate(ids):
            if not len(self.allframes[ctid - self.first_tid]):
                with self.zip_file.open(self.sid + '_Hour_' + str(ctid) + '_Band_09.png') as file:
                    frame = np.array(Image.open(io.BytesIO(file.read()))) / 255.0 
                    frame = frame[:, :, np.newaxis]
                    frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])  
                    self.allframes[ctid - self.first_tid] = frame
            else:
                frame = self.allframes[ctid - self.first_tid]

            frames.append(frame)

        return ids, frames