```python
train_ft['일시'] = train_ft['일시'].apply(lambda x: int(datetime.strptime(str(x), "%Y-%m-%d").timestamp()))
train_ft
```
