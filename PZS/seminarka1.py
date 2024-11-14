import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

data_path = "PZS/data/100001_ECG.dat"
hz = 1000
seconds = 100

def load_data(data_path):
    data = np.fromfile(data_path, dtype=np.int16)
    return data

def load_annotated(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                annotations.append([int(x) if x.strip() else 0 for x in line.split(',')])
    annotations = np.array(annotations)
    consensus_annotations = annotations[:, :3]
    annotator1_annotations = annotations[:, 3:6]
    annotator2_annotations = annotations[:, 6:9]
    annotator3_annotations = annotations[:, 9:12]

    return consensus_annotations, annotator1_annotations, annotator2_annotations, annotator3_annotations

con, a1, a2, a3 = load_annotated("PZS/data/100001_ANN.csv")

d = load_data(data_path)


d = d[:hz*seconds]
print(d)

plt.plot(d)
d = d - np.mean(d)

sos = signal.butter(1, 0.5, 'hp', fs=hz, output='sos')
d = signal.sosfilt(sos, d)

vrcholy, _ = signal.find_peaks(d, height=1200)

min_vzdalenost = int(0.5 * hz)
filtrovane_vrcholy = []
posledni_vrchol = -min_vzdalenost

for vrchol in vrcholy:
    if vrchol - posledni_vrchol >= min_vzdalenost:
        filtrovane_vrcholy.append(vrchol)
        posledni_vrchol = vrchol

beatu = len(filtrovane_vrcholy)
minuty = seconds / 60
pulses_per_minute = beatu / minuty

print(f"BPM: {pulses_per_minute:.2f}")

for annotation in con:
    start, end, quality = annotation
    if start < hz * seconds and end < hz * seconds:
        if quality == 1:
            color = 'green'
        elif quality == 2:
            color = 'yellow'
        elif quality == 3:
            color = 'red'
        else:
            color = 'blue'  # Default color for unexpected quality values
        plt.axvspan(start, end, color=color, alpha=0.3)

plt.plot(d)
plt.plot(filtrovane_vrcholy, d[filtrovane_vrcholy], "x")
plt.title("ECG Signál s Detekovanými Vrcholy")
plt.xlabel("Číslo Vzorku")
plt.ylabel("Amplituda")
plt.show()

