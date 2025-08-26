
Running the code below in Browser (Woooooowwwwww!!!!!!). [^1]


``` pyodide install="pandas,ssl"
import ssl
import pandas as pd

df = pd.DataFrame()
df['AAPL'] = pd.Series([1, 2, 3])
df['MSFT'] = pd.Series([4, 5, 6])
df['GOOGL'] = pd.Series([7, 8, 9])

print(df)

```

[^1]: [Pyodide](https://pawamoy.github.io/markdown-exec/usage/pyodide/){target="_blank"}



