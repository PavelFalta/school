import wfdb
record = wfdb.rdsamp('PZS/data/16265', sampto=3000)
annotation = wfdb.rdann('PZS/data/16265', 'atr', sampto=3000)

print(record[1])