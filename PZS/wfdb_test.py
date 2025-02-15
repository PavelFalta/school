import wfdb
path = 'PZS/data/mit-bih-normal-sinus-rhythm-database-1.0.0/16420'
record = wfdb.rdsamp(path, sampto=3000)
annotation = wfdb.rdann(path, 'atr', sampto=3000)

print(record)