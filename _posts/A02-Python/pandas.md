设置pandas参数

```python
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
```

[Applying pandas qcut bins to new data](https://stackoverflow.com/questions/37906210/applying-pandas-qcut-bins-to-new-data)

```python


```





groupby之后获取组名或某一组的数据

```python
orders_grp = orders.groupby("user_id")
grp1 = orders_grp.get_group(1001)

orders_grp.groups
```



loc, iloc，[]的区别	



https://blog.csdn.net/Xw_Classmate/article/details/51333646